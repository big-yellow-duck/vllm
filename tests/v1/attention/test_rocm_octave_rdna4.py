# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving adapter: packed stores, mixed batches and captured BF16/FP16 output."""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tests.utils import RemoteOpenAIServer
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.attention.backends.rocm_octave_attn import RocmOctaveMetadata
from vllm.v1.attention.backends.rocm_octave_rdna4_attn import (
    RocmOctaveRdna4AttentionBackend,
    RocmOctaveRdna4AttentionImpl,
    RocmOctaveRdna4MetadataBuilder,
)
from vllm.v1.attention.ops import rocm_octave as sq
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    reset_workspace_manager,
)


def _rdna4():
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_rdna4

    return on_rdna4()


@pytest.fixture(scope="module")
def config():
    return EngineArgs(
        model="Qwen/Qwen3.8-27B-FP8",
        tensor_parallel_size=2,
        dtype="bfloat16",
        kv_cache_dtype="octave_k3v4",
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        language_model_only=True,
        enforce_eager=True,
    ).create_engine_config()


@pytest.mark.skipif(not _rdna4(), reason="Native RDNA4 selection requires gfx1200/1")
def test_octave_cache_selects_rdna4(config):
    from vllm.v1.attention.selector import get_attn_backend

    with set_current_vllm_config(config):
        backend = get_attn_backend(256, torch.bfloat16, "octave_k3v4")
    assert backend is RocmOctaveRdna4AttentionBackend


@pytest.mark.skipif(not _rdna4(), reason="Native RDNA4 adapter requires gfx1200/1")
@pytest.mark.parametrize(
    "cache_dtype", ["octave_k3v3", "octave_k3v4", "octave_k3v3_compact"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("mode", ["decode", "prefill", "mixed"])
def test_serving_store_and_graph_replay(config, cache_dtype, dtype, token_major, mode):
    """Use real backend/config objects and compare every row to packed-KV Torch."""
    device = torch.device("cuda", 0)
    torch.manual_seed(1171)
    head, heads, kv_heads, page = 256, 12, 2, 16
    fmt = sq.OctaveFormat.from_cache_dtype(cache_dtype, head, 64)
    with set_current_vllm_config(config):
        impl = RocmOctaveRdna4AttentionImpl(
            heads, head, head**-0.5, num_kv_heads=kv_heads, kv_cache_dtype=cache_dtype
        )
    assert RocmOctaveRdna4AttentionBackend.supports_head_size(head)
    counts = (
        [1, 1] if mode == "decode" else [160, 33] if mode == "prefill" else [1, 160, 33]
    )
    prefixes = (
        [17, 49] if mode == "decode" else [0, 65] if mode == "prefill" else [17, 0, 65]
    )
    lengths = [p + c for p, c in zip(prefixes, counts)]
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    n = starts[-1]
    pages = sum(math.ceil(length / page) for length in lengths)
    shape = (
        (pages + 2, page, kv_heads, fmt.slot_bytes)
        if token_major
        else (pages + 2, kv_heads, page, fmt.slot_bytes)
    )
    cache = torch.full(shape, 239, dtype=torch.uint8, device=device)
    if token_major:
        cache = cache.transpose(1, 2)
    table = torch.full(
        (len(counts), math.ceil(max(lengths) / page) + 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    physical = torch.randperm(pages, device=device) + 1
    keys, values, slots = [], [], []
    offset = 0
    for req, (prefix, count, length) in enumerate(zip(prefixes, counts, lengths)):
        size = math.ceil(length / page)
        table[req, :size] = physical[offset : offset + size]
        offset += size
        k = torch.randn(length, kv_heads, head, device=device, dtype=dtype)
        v = torch.randn_like(k)
        pos = torch.arange(length, device=device)
        if prefix:
            cache[table[req, pos[:prefix] // page].long(), :, pos[:prefix] % page] = (
                sq.reference_quantize(k[:prefix], v[:prefix], fmt)
            )
        keys.append(k[prefix:])
        values.append(v[prefix:])
        slots.append(
            table[req, pos[prefix:] // page].long() * page + pos[prefix:] % page
        )
    key, value = torch.cat(keys), torch.cat(values)
    slot_mapping = torch.cat(slots)
    untouched = cache.clone()
    unwritten = torch.ones(pages + 2, page, dtype=torch.bool, device=device)
    unwritten[slot_mapping // page, slot_mapping % page] = False
    query = torch.randn(n, heads, head, device=device, dtype=dtype)
    output = torch.empty_like(query)
    nd = len(counts) if mode == "decode" else 1 if mode == "mixed" else 0
    md = RocmOctaveMetadata(
        num_actual_tokens=n,
        max_query_len=max(counts),
        query_start_loc=torch.tensor(starts, device=device, dtype=torch.int32),
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
        block_table=table,
        slot_mapping=slot_mapping,
        num_decodes=nd,
        num_decode_tokens=nd,
        q_to_req=torch.empty(n, device=device, dtype=torch.int32),
        q_to_klen=torch.empty(n, device=device, dtype=torch.int32),
    )
    RocmOctaveRdna4MetadataBuilder._fill_query_maps(md)

    def call():
        impl.do_kv_cache_update(
            None, key.flatten(1), value.flatten(1), cache, slot_mapping
        )
        return impl.forward(
            None,
            query.flatten(1),
            key.flatten(1),
            value.flatten(1),
            cache,
            md,
            output=output.flatten(1),
        )

    def reference():
        result = torch.empty(query.shape, device=device, dtype=torch.float32)
        for req, (count, prefix, length) in enumerate(zip(counts, prefixes, lengths)):
            lo, hi = starts[req : req + 2]
            pos = torch.arange(length, device=device)
            k, v = sq.reference_dequantize(
                cache[table[req, pos // page].long(), :, pos % page], fmt
            )
            if req >= nd and mode != "decode":
                k = torch.cat((k[:prefix], key[lo:hi].float()))
                v = torch.cat((v[:prefix], value[lo:hi].float()))
            k, v = (
                k.repeat_interleave(heads // kv_heads, dim=1),
                v.repeat_interleave(heads // kv_heads, dim=1),
            )
            score = torch.einsum("qhd,khd->hqk", query[lo:hi].float(), k) * head**-0.5
            causal = (
                pos[None, :] > (prefix + torch.arange(count, device=device))[:, None]
            )
            weights = score.masked_fill(causal[None], -torch.inf).softmax(-1)
            result[lo:hi] = torch.einsum("hqk,khd->qhd", weights, v)
        return result

    init_workspace_manager(device)
    try:
        impl.forward(
            None,
            query.flatten(1),
            key.flatten(1),
            value.flatten(1),
            cache,
            None,
            output=output.flatten(1),
        )
        call()
        stored = cache[slot_mapping // page, :, slot_mapping % page]
        actual_k, actual_v = sq.reference_dequantize(stored, fmt)
        expected_k, expected_v = sq.reference_dequantize(
            sq.reference_quantize(key, value, fmt), fmt
        )
        for actual, expected in ((actual_k, expected_k), (actual_v, expected_v)):
            assert ((actual - expected).norm() / expected.norm()).item() < 0.01
        current_workspace_manager().lock()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        query.neg_()
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        expected = reference()
        error = (output.float() - expected).norm(dim=-1) / expected.norm(
            dim=-1
        ).clamp_min(1e-20)
        assert torch.isfinite(output).all()
        assert error.max().item() < 0.02
        assert torch.equal(
            cache.transpose(1, 2)[unwritten], untouched.transpose(1, 2)[unwritten]
        )
    finally:
        reset_workspace_manager()


@pytest.mark.distributed
@pytest.mark.slow_test
@pytest.mark.skipif(not _rdna4(), reason="Native RDNA4 serving requires gfx1200/1")
def test_qwen27_fp8_tp2_serving(tmp_path):
    """Real TP2 server: greedy answers, batched decode, and cached-prefix recall."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("TP2 serving requires two RDNA4 GPUs")
    model = "Qwen/Qwen3.8-27B-FP8"
    args = [
        "--revision",
        "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "--language-model-only",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--attention-backend",
        "ROCM_OCTAVE_RDNA4",
        "--kv-cache-dtype",
        "octave_k3v4",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.9",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--generation-config",
        "vllm",
        "--compilation-config",
        (
            '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
            '"cudagraph_capture_sizes":[1,2,4]}'
        ),
    ]
    records = []
    with RemoteOpenAIServer(model, args, max_wait_seconds=1200) as server:
        client = server.get_client()

        def answer(prompt):
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Give only the answer. No explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=32,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = result.choices[0].message.content
            assert content is not None and content.strip()
            return result.model_dump()

        prompts = [
            "What is 2 + 2?",
            "What is 3 + 5?",
            "What is 7 - 2?",
            "What is 4 * 3?",
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(answer, prompts))
        for prompt, expected, response in zip(prompts, ["4", "8", "5", "12"], results):
            assert response["choices"][0]["message"]["content"].strip() == expected
            records.append(dict(prompt=prompt, response=response))
        prefix = (
            "The access code is ORCHID-729.\n"
            + "This is neutral padding for the cache recall check.\n" * 512
        )
        prompt = prefix + "\nWhat is the access code? Give just the code."
        for _ in range(2):
            response = answer(prompt)
            assert "ORCHID-729" in response["choices"][0]["message"]["content"]
            records.append(dict(prompt=prompt, response=response))
        cached = records[-1]["response"]["usage"].get("prompt_tokens_details")
        assert cached is not None and cached["cached_tokens"] > 0
    (tmp_path / "responses.json").write_text(json.dumps(records, indent=2) + "\n")
    print("TP2 serving responses:", tmp_path / "responses.json")
