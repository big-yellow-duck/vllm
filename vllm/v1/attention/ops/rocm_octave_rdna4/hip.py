# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and call the standalone RDNA4 HIP Octave kernels on a Torch stream."""

import ctypes as ct
import fcntl
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parent
FORMATS = {
    "octave_k3v3": (3, 216),
    "octave_k3v4": (4, 248),
    "octave_k3v3_compact": (19, 196),
}
DTYPES = {torch.float16: 0, torch.bfloat16: 1}
HEAD = 256


def cache_root():
    from vllm import envs

    return Path(envs.VLLM_CACHE_ROOT) / "octave_rdna4"


def build_library(source_dir=ROOT / "native/legacy", build_dir=None, arch="gfx1201"):
    """Build a library keyed by source and flags to isolate candidates."""
    if arch not in ("gfx1200", "gfx1201"):
        raise ValueError(f"Unsupported RDNA4 target {arch}")
    source_dir = Path(source_dir)
    build_dir = cache_root() / "build" if build_dir is None else Path(build_dir)
    hipcc = shutil.which("hipcc") or str(Path(sys.executable).with_name("hipcc"))
    flags = [f"--offload-arch={arch}", "-std=c++17", "-O3", "-fPIC", "-shared"]
    digest = hashlib.sha256(repr([hipcc, flags]).encode())
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(source_dir)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / f"octave-{arch}-{digest.hexdigest()[:20]}.so"
    with library.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with TemporaryDirectory(dir=build_dir) as temp:
                target = Path(temp) / library.name
                subprocess.run(
                    [
                        hipcc,
                        "-x",
                        "hip",
                        *flags,
                        str(source_dir / "octave_rdna4.hip"),
                        "-o",
                        str(target),
                    ],
                    check=True,
                )
                target.replace(library)
    return library.resolve()


class _Activation(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("token_stride", ct.c_int64),
        ("head_stride", ct.c_int64),
    ]


class _PagedCache(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("page_stride", ct.c_int64),
        ("head_stride", ct.c_int64),
        ("token_stride", ct.c_int64),
        ("page_size", ct.c_int),
        ("heads", ct.c_int),
    ]


class _Shadow(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("tags", ct.c_void_p),
        ("locks", ct.c_void_p),
        ("entries", ct.c_int),
    ]


def _shadow(table, tags, locks, device, dtype, heads, head_dim):
    entries = table.shape[0]
    if (
        entries <= 0
        or entries & (entries - 1)
        or table.shape != (entries, heads, 2 * head_dim)
        or table.dtype != dtype
        or table.device != device
        or not table.is_contiguous()
    ):
        raise ValueError("Expected a contiguous power-of-two exact K/V table")
    return _Shadow(
        table.data_ptr(),
        _vector(tags, entries, torch.int32, device),
        _vector(locks, entries, torch.int32, device) if locks is not None else None,
        entries,
    )


def _activation(x, device, dtype, head_dim=HEAD):
    if (
        x.ndim != 3
        or x.shape[-1] != head_dim
        or x.stride(-1) != 1
        or x.device != device
        or x.dtype != dtype
        or dtype not in DTYPES
    ):
        raise ValueError(
            f"Expected (tokens, heads, {head_dim}) FP16 or BF16 on one ROCm device"
        )
    return _Activation(x.data_ptr(), x.stride(0), x.stride(1))


def _cache_view(cache, cache_dtype, device, layout=None):
    code, bytes_per_slot = FORMATS[cache_dtype]
    if layout is not None:
        code, bytes_per_slot = layout.kernel_code, layout.slot_bytes
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 4
        or cache.shape[3] != bytes_per_slot
        or cache.stride(3) != 1
        or cache.device != device
        or cache.shape[1] == 0
        or cache.shape[2] == 0
        or cache.data_ptr() % 4
        or any(stride % 4 for stride in cache.stride()[:3])
    ):
        raise ValueError(
            "Expected aligned uint8 cache (pages, KV heads, page size, slot)"
        )
    return _PagedCache(
        cache.data_ptr(), *cache.stride()[:3], cache.shape[2], cache.shape[1]
    ), code


def _vector(tensor, count, dtype, device):
    if (
        tensor.shape != (count,)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"Expected a contiguous vector of {count} {dtype} entries")
    return tensor.data_ptr()


def _check_buffer(tensor, shape, device):
    if (
        tensor.shape != shape
        or tensor.dtype != torch.float32
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"Expected a contiguous FP32 buffer of shape {shape}")
    return tensor.data_ptr()


class HipOctave:
    """Paged Octave on gfx1200/gfx1201, with FP32 attention output.

    Callers supply valid page indices and unique live write slots. Q and K are
    post-RoPE. The contiguous int32 sign words follow the Octave format.
    """

    def __init__(self, library):
        if not torch.version.hip:
            raise RuntimeError("RDNA4 Octave requires PyTorch built for ROCm")
        self._library = ct.CDLL(str(Path(library).resolve()))
        from .geometry import OctaveGeometry

        self.head_dim, self.rope_dim, self.geometry_version = HEAD, 64, 0
        if hasattr(self._library, "octave_rdna4_geometry_version"):
            for name in ("head_size", "rope_dim", "geometry_version"):
                getter = getattr(self._library, "octave_rdna4_" + name)
                getter.argtypes, getter.restype = [], ct.c_int
                setattr(self, {"head_size": "head_dim"}.get(name, name), getter())
        self.geometry = OctaveGeometry(self.head_dim, self.rope_dim)
        ptr, i32, i64, f32 = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_float
        self._store = self._bind(
            "store",
            [_Activation, _Activation, _PagedCache, ptr, ptr, i32, i32, i32, ptr],
        )
        common = [ptr, ptr, _Activation]
        meta = [_PagedCache, ptr, i64, ptr, ptr]
        tail = [ptr, i32, i32, i32, f32, i32, i32, ptr]
        self._decode = self._bind("decode", common + meta + tail)
        self._prefill = self._bind(
            "prefill", common + [_Activation, _Activation] + meta + [i32] + tail
        )
        self._window_store = self._sparse = None
        if self.geometry_version:
            self._window_store = self._bind(
                "window_store",
                [
                    _Activation,
                    _Activation,
                    ptr,
                    ptr,
                    _Shadow,
                    _Shadow,
                    i32,
                    i32,
                    i32,
                    ptr,
                ],
            )
            self._sparse = self._bind(
                "decode_sparse",
                common
                + [
                    _PagedCache,
                    ptr,
                    i64,
                    i32,
                    i32,
                    i32,
                    ptr,
                    ptr,
                    i64,
                    i32,
                    ptr,
                    ptr,
                    _Shadow,
                    _Shadow,
                ]
                + tail[1:],
            )

    def layout(self, cache_dtype):
        """Wire layout, logical/storage dimensions and CPU sign words."""
        return self.geometry.layout(cache_dtype)

    def _bind(self, name, argtypes):
        fn = getattr(self._library, f"octave_rdna4_{name}")
        fn.argtypes = argtypes
        fn.restype = ct.c_int
        return fn

    @staticmethod
    def _launch(fn, device, *args):
        if device.type != "cuda":
            raise ValueError("RDNA4 Octave requires GPU tensors")
        with torch.accelerator.device_index(device.index):
            status = fn(*args, torch.cuda.current_stream(device).cuda_stream)
        if status:
            raise RuntimeError(f"{fn.__name__} failed with HIP status {status}")

    def store(self, key, value, cache, slots, signs, cache_dtype):
        device, dtype = key.device, key.dtype
        k, v = (
            _activation(key, device, dtype, self.head_dim),
            _activation(value, device, dtype, self.head_dim),
        )
        pages, code = _cache_view(cache, cache_dtype, device, self.layout(cache_dtype))
        if (
            key.shape != value.shape
            or key.shape[1] != pages.heads
            or slots.numel() > key.shape[0]
        ):
            raise ValueError("Cache head count or slot count does not match K/V")
        n = slots.numel()
        self._launch(
            self._store,
            device,
            k,
            v,
            pages,
            _vector(slots, n, torch.int64, device),
            _vector(
                signs,
                self.layout(cache_dtype).storage_head_dim // 32,
                torch.int32,
                device,
            ),
            n,
            DTYPES[dtype],
            code,
        )

    def window_store(
        self, key, value, slots, positions, wtab, wtags, wlocks, stab, stags, slocks
    ):
        """Keep exact K/V for selected recent and leading QSA positions.

        Direct-mapped tables use physical-slot tags. Collisions evict complete
        rows, so sparse attention can safely fall back to packed KV.
        """
        if self._window_store is None:
            raise RuntimeError("Sparse Octave requires the geometry runtime")
        if wlocks is None or slocks is None:
            raise ValueError("Exact K/V stores require both entry-lock tables")
        device, dtype = key.device, key.dtype
        k, v = (_activation(x, device, dtype, self.head_dim) for x in (key, value))
        n, heads = slots.numel(), key.shape[1]
        if key.shape != value.shape or n > key.shape[0]:
            raise ValueError("Exact K/V rows and slot counts must match")
        recent = _shadow(wtab, wtags, wlocks, device, dtype, heads, self.head_dim)
        sinks = _shadow(stab, stags, slocks, device, dtype, heads, self.head_dim)
        self._launch(
            self._window_store,
            device,
            k,
            v,
            _vector(slots, n, torch.int64, device),
            _vector(positions, n, torch.int32, device),
            recent,
            sinks,
            n,
            heads,
            DTYPES[dtype],
        )
        stream = torch.cuda.current_stream(device)
        for tensor in (
            key,
            value,
            slots,
            positions,
            wtab,
            wtags,
            wlocks,
            stab,
            stags,
            slocks,
        ):
            tensor.record_stream(stream)

    def decode_sparse(
        self,
        query,
        cache,
        block_table,
        q_to_req,
        indices,
        signs,
        cache_dtype,
        *,
        positions=None,
        wtab=None,
        wtags=None,
        stab=None,
        stags=None,
        scale=None,
        splits=1,
        out=None,
        scratch=None,
    ):
        """FP32 sparse attention over selected logical positions and holes.

        Optional exact tables preserve the first four and last 64 selected
        positions. Missing or evicted tags use the packed cache. No prefix is
        expanded to a dense K/V tensor. All buffers may be supplied for capture.
        """
        if self._sparse is None:
            raise RuntimeError("Sparse Octave requires the geometry runtime")
        device, dtype = query.device, query.dtype
        q = _activation(query, device, dtype, self.head_dim)
        tokens, heads, _ = query.shape
        pages, code = _cache_view(cache, cache_dtype, device, self.layout(cache_dtype))
        if heads <= 0 or heads % pages.heads or not 1 <= splits <= 1024:
            raise ValueError("Invalid sparse GQA group or split count")
        if (
            block_table.ndim != 2
            or block_table.stride(1) != 1
            or block_table.dtype != torch.int32
            or block_table.device != device
            or indices.ndim != 2
            or indices.shape[0] != tokens
            or indices.stride(1) != 1
            or indices.dtype != torch.int32
            or indices.device != device
        ):
            raise ValueError("Expected int32 page tables and sparse index rows")
        position_ptr, recent, sinks = None, _Shadow(), _Shadow()
        if positions is not None:
            if any(x is None for x in (wtab, wtags, stab, stags)):
                raise ValueError("Exact sparse attention requires both tagged tables")
            position_ptr = _vector(positions, tokens, torch.int32, device)
            recent = _shadow(
                wtab, wtags, None, device, dtype, pages.heads, self.head_dim
            )
            sinks = _shadow(
                stab, stags, None, device, dtype, pages.heads, self.head_dim
            )
        if out is None:
            out = torch.empty_like(query, dtype=torch.float32)
        output_ptr = _check_buffer(out, query.shape, device)
        partial_ptr = None
        if splits > 1:
            shape = (tokens, heads, splits, self.head_dim + 2)
            if scratch is None:
                scratch = torch.empty(shape, dtype=torch.float32, device=device)
            partial_ptr = _check_buffer(scratch, shape, device)
        self._launch(
            self._sparse,
            device,
            output_ptr,
            partial_ptr,
            q,
            pages,
            block_table.data_ptr(),
            block_table.stride(0),
            block_table.shape[1],
            block_table.shape[0],
            cache.shape[0],
            _vector(q_to_req, tokens, torch.int32, device),
            indices.data_ptr(),
            indices.stride(0),
            indices.shape[1],
            _vector(
                signs,
                self.layout(cache_dtype).storage_head_dim // 32,
                torch.int32,
                device,
            ),
            position_ptr,
            recent,
            sinks,
            tokens,
            heads,
            splits,
            self.head_dim**-0.5 if scale is None else scale,
            DTYPES[dtype],
            code,
        )
        stream = torch.cuda.current_stream(device)
        for tensor in (
            query,
            cache,
            block_table,
            q_to_req,
            indices,
            signs,
            positions,
            wtab,
            wtags,
            stab,
            stags,
            out,
            scratch,
        ):
            if tensor is not None:
                tensor.record_stream(stream)
        return out

    def decode(
        self,
        query,
        cache,
        block_table,
        q_to_req,
        q_to_klen,
        signs,
        cache_dtype,
        *,
        scale=None,
        splits=1,
        out=None,
        scratch=None,
    ):
        return self._attention(
            query,
            cache,
            block_table,
            q_to_req,
            q_to_klen,
            signs,
            cache_dtype,
            scale,
            splits,
            out,
            scratch,
        )

    def prefill(
        self,
        query,
        key,
        value,
        cache,
        block_table,
        query_start_loc,
        seq_lens,
        signs,
        cache_dtype,
        *,
        scale=None,
        splits=1,
        out=None,
        scratch=None,
    ):
        return self._attention(
            query,
            cache,
            block_table,
            query_start_loc,
            seq_lens,
            signs,
            cache_dtype,
            scale,
            splits,
            out,
            scratch,
            key=key,
            value=value,
        )

    def _attention(
        self,
        query,
        cache,
        table,
        mapping,
        lengths,
        signs,
        cache_dtype,
        scale,
        splits,
        out,
        scratch,
        *,
        key=None,
        value=None,
    ):
        device, dtype = query.device, query.dtype
        if scale is None:
            scale = self.head_dim**-0.5
        q = _activation(query, device, dtype, self.head_dim)
        tokens, heads, _ = query.shape
        pages, code = _cache_view(cache, cache_dtype, device, self.layout(cache_dtype))
        if heads == 0 or heads % pages.heads or not 1 <= splits <= 256:
            raise ValueError("Invalid GQA group or split count")
        if (
            table.ndim != 2
            or table.dtype != torch.int32
            or table.device != device
            or table.stride(1) != 1
        ):
            raise ValueError("Expected an int32 page table with contiguous rows")
        if out is None:
            out = torch.empty(query.shape, device=device, dtype=torch.float32)
        out_ptr = _check_buffer(out, query.shape, device)
        partials = None
        if splits > 1:
            shape = (tokens, heads, splits, self.head_dim + 2)
            if scratch is None:
                scratch = torch.empty(shape, device=device, dtype=torch.float32)
            partials = _check_buffer(scratch, shape, device)
        meta = (pages, table.data_ptr(), table.stride(0))
        tail = (
            _vector(
                signs,
                self.layout(cache_dtype).storage_head_dim // 32,
                torch.int32,
                device,
            ),
            tokens,
            heads,
            splits,
            scale,
            DTYPES[dtype],
            code,
        )
        if key is None:
            self._launch(
                self._decode,
                device,
                out_ptr,
                partials,
                q,
                *meta,
                _vector(mapping, tokens, torch.int32, device),
                _vector(lengths, tokens, torch.int32, device),
                *tail,
            )
        else:
            k, v = (
                _activation(key, device, dtype, self.head_dim),
                _activation(value, device, dtype, self.head_dim),
            )
            reqs = lengths.numel()
            if (
                key.shape != (tokens, pages.heads, self.head_dim)
                or value.shape != key.shape
            ):
                raise ValueError("Expected one unquantized chunk K/V row per query")
            if reqs != table.shape[0] or (tokens > 0 and reqs == 0):
                raise ValueError("Prefill sequence lengths must match the page table")
            self._launch(
                self._prefill,
                device,
                out_ptr,
                partials,
                q,
                k,
                v,
                *meta,
                _vector(mapping, reqs + 1, torch.int32, device),
                _vector(lengths, reqs, torch.int32, device),
                reqs,
                *tail,
            )
        return out
