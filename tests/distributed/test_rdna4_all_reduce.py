# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import vllm
from vllm.distributed.device_communicators import rdna4_all_reduce as rdna4_module
from vllm.distributed.device_communicators.rdna4_all_reduce import (
    HOST_ALL_REDUCE_MAX_SIZE,
    P2P_GRAPH_MIN_SIZE,
    TP8_PYNCCL_GRAPH_RANGE,
    RDNA4AllReduce,
)


class _Delegate:
    def __init__(self, accepts, result):
        self.accepts = accepts
        self.result = result
        self.should_use_calls = 0
        self.all_reduce_calls = 0

    def should_use(self, _tensor):
        self.should_use_calls += 1
        return self.accepts

    def all_reduce(self, *_args, **_kwargs):
        self.all_reduce_calls += 1
        return self.result


def _bare_communicator(world_size, *, max_size=2 * 1024 * 1024):
    communicator = RDNA4AllReduce.__new__(RDNA4AllReduce)
    communicator.disabled = False
    communicator.world_size = world_size
    communicator.rank = 0
    communicator.device = torch.device("cpu")
    communicator.max_size = max_size
    communicator._tp2_comm = None
    communicator._host_comm = None
    communicator._p2p_ready = True
    communicator._IS_CAPTURING = False
    communicator.fully_connected = True
    return communicator


def _bf16_tensor(nbytes):
    assert nbytes % torch.bfloat16.itemsize == 0
    return torch.empty(nbytes // torch.bfloat16.itemsize, dtype=torch.bfloat16)


def test_transport_priority_is_tp2_then_host_then_p2p():
    tensor = _bf16_tensor(64)
    communicator = _bare_communicator(2)
    tp2 = _Delegate(True, "tp2")
    host = _Delegate(True, "host")
    communicator._tp2_comm = tp2
    communicator._host_comm = host
    communicator._should_use_p2p = MagicMock(return_value=True)

    assert communicator.custom_all_reduce(tensor) == "tp2"
    assert tp2.all_reduce_calls == 1
    assert host.should_use_calls == 0
    communicator._should_use_p2p.assert_not_called()

    communicator._tp2_comm = _Delegate(False, None)
    assert communicator.custom_all_reduce(tensor) == "host"
    assert host.all_reduce_calls == 1
    communicator._should_use_p2p.assert_not_called()


@pytest.mark.parametrize(
    ("world_size", "nbytes", "expected"),
    [
        (4, P2P_GRAPH_MIN_SIZE[4] - 64, False),
        (4, P2P_GRAPH_MIN_SIZE[4], True),
        (8, HOST_ALL_REDUCE_MAX_SIZE[8], False),
        (8, HOST_ALL_REDUCE_MAX_SIZE[8] + 64, True),
        (8, TP8_PYNCCL_GRAPH_RANGE[0], False),
        (8, TP8_PYNCCL_GRAPH_RANGE[1], False),
        (8, TP8_PYNCCL_GRAPH_RANGE[1] + 64, True),
    ],
)
def test_p2p_policy_boundaries(world_size, nbytes, expected):
    communicator = _bare_communicator(world_size, max_size=2 * 1024 * 1024)
    tensor = _bf16_tensor(nbytes)
    assert communicator._p2p_tensor_supported(tensor) is expected
    assert communicator.should_use_graph(tensor) is expected
    assert not communicator.should_use(tensor)


@pytest.mark.parametrize("world_size", [4, 8])
def test_graph_dispatches_to_tp_specific_p2p_kernel(monkeypatch, world_size):
    communicator = _bare_communicator(world_size)
    communicator._IS_CAPTURING = True
    communicator._captured_outputs = []
    communicator._capture_base_index = 0
    communicator._gpu_graph_output_ptrs_array = torch.zeros((1, 8), dtype=torch.int64)
    communicator._should_use_p2p = MagicMock(return_value=True)
    communicator._run_p2p_tp4_push_rsag = MagicMock(
        side_effect=lambda _, out, **__: out
    )
    communicator._run_p2p_hierarchical_tp8 = MagicMock(
        side_effect=lambda _, out, **__: out
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    tensor = _bf16_tensor(1024 * 1024)
    output = torch.empty_like(tensor)

    assert communicator.custom_all_reduce(tensor, out=output) is output
    if world_size == 4:
        communicator._run_p2p_tp4_push_rsag.assert_called_once()
        communicator._run_p2p_hierarchical_tp8.assert_not_called()
    else:
        communicator._run_p2p_hierarchical_tp8.assert_called_once()
        communicator._run_p2p_tp4_push_rsag.assert_not_called()


@pytest.mark.parametrize(
    ("numel", "kind", "blocks"),
    [
        (65_536, "full", 1),
        (65_544, "full", 2),
        (196_608, "full", 2),
        (196_616, "full", 4),
        (786_432, "full", 4),
        (786_440, "pipeline", 8),
        (2_097_160, "pipeline", 10),
        (3_801_096, "pipeline", 12),
        (5_242_888, "pipeline", 16),
    ],
)
def test_tp2_policy_boundaries(numel, kind, blocks):
    pytest.importorskip("flydsl")
    from vllm.distributed.device_communicators.rdna4_all_reduce_tp2 import (
        RDNA4TP2AllReduce,
    )

    selected_kind, selected_blocks, chunk_packs = RDNA4TP2AllReduce._policy(numel, 1024)
    assert (selected_kind, selected_blocks) == (kind, blocks)
    assert chunk_packs == 0 if kind == "full" else chunk_packs > 0


def test_host_transport_selects_direct_then_rsag(monkeypatch):
    pytest.importorskip("flydsl")
    import vllm.distributed.device_communicators.rdna4_all_reduce_host as host_module
    from vllm.distributed.device_communicators.rdna4_all_reduce_host import (
        RDNA4HostAllReduce,
    )

    communicator = RDNA4HostAllReduce.__new__(RDNA4HostAllReduce)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.max_size = HOST_ALL_REDUCE_MAX_SIZE[8]
    communicator.world_size = 8
    communicator.rank = 0
    communicator.threads = 512
    communicator.pipeline_blocks = 16
    communicator.pipeline_min_numel = 65_536
    communicator._device_address = 0x1000
    communicator._slot_bytes = 384 * 1024
    communicator._full_launcher = MagicMock()
    rsag_launcher = MagicMock()
    monkeypatch.setattr(host_module, "Int32", lambda value: value)
    monkeypatch.setattr(host_module, "Int64", lambda value: value)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    monkeypatch.setattr(
        host_module, "make_host_rsag_launcher", lambda **_kwargs: rsag_launcher
    )

    communicator.all_reduce(_bf16_tensor((65_536 - 8) * 2))
    communicator._full_launcher.assert_called_once()
    rsag_launcher.assert_not_called()

    communicator.all_reduce(_bf16_tensor(65_536 * 2))
    rsag_launcher.assert_called_once()


def test_missing_flydsl_disables_before_gpu_allocation(monkeypatch):
    monkeypatch.setattr(rdna4_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rdna4_module.dist, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(rdna4_module.dist, "get_world_size", lambda _group: 4)
    monkeypatch.setattr(rdna4_module.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(rdna4_module.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna4", lambda: True)
    monkeypatch.setattr(rdna4_module, "_is_rdna4_flydsl_available", lambda: False)
    get_properties = MagicMock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_properties)

    communicator = RDNA4AllReduce(object(), torch.device("cuda:0"))

    assert communicator.disabled
    get_properties.assert_not_called()


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_flydsl_probe_fails_closed(monkeypatch, error):
    def unavailable(_name):
        raise error("FlyDSL is unavailable")

    monkeypatch.setattr(rdna4_module.importlib, "import_module", unavailable)
    assert not rdna4_module._is_rdna4_flydsl_available()


def test_vllm_kernels_do_not_import_private_flydsl_mlir():
    kernel_dir = (
        Path(vllm.__file__).parent
        / "distributed"
        / "device_communicators"
        / "flydsl_kernels"
    )
    for path in kernel_dir.glob("rdna4_all_reduce_*.py"):
        tree = ast.parse(path.read_text())
        private_imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("flydsl._mlir")
        ]
        assert private_imports == [], f"{path.name}: {private_imports}"
