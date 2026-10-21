# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for segmented conversion selection and startup tuning."""

import itertools
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import (
    rocm_segmented_attn_autotune_warmup as tuning,
)
from vllm.v1.attention.ops import segmented_attention as segmented
from vllm.v1.attention.ops import segmented_attention_config as selection


def test_tuning_table_stride_is_fixed_without_extending_the_workload():
    """Reuse compiled launches across sequence buckets while timing actual lengths."""
    for seq_len in (128, 512):
        inputs = tuning._make_inputs(
            "cpu",
            torch.bfloat16,
            torch.bfloat16,
            2,
            1,
            64,
            32,
            64,
            (2, 16, seq_len),
            table_max_len=1024,
        )
        assert inputs[5].shape == (2, 32)
        assert inputs[5].stride(0) == 32
        assert inputs[7].tolist() == [seq_len, seq_len]
        assert inputs[3].shape[0] == 2 * (seq_len // 32)


@pytest.mark.parametrize(
    "mode,image_span,expected",
    [
        ("none", 1089, 1279),
        ("clamped", 1089, 2367),
        ("unclamped", 2048, 4350),
        ("clamped", 0, 32768),
    ],
)
def test_mm_span_includes_reopened_image_keys(mode, image_span, expected):
    """Long text prefixes must not inflate a windowed image workload's picker."""
    assert (
        selection.segmented_attention_span(32768, 256, 1023, mode, image_span)
        == expected
    )
    assert selection.segmented_attention_span(32768, 256, -1, mode, image_span) == 32768


def test_mm_tuning_keys_isolate_masks_image_spans_and_range_counts():
    """Neither text nor incompatible image geometry may supply an MM winner."""
    args = ("cuda:0", torch.bfloat16, torch.float8_e4m3fn, 16, 8, 256, 32, 1.0)
    text = tuning._key(*args)
    image = tuning._key(
        *args, mm_prefix_mode="clamped", mm_prefix_max_span=1089, mm_prefix_max_ranges=1
    )
    assert image == tuning._key(
        *args, mm_prefix_mode="clamped", mm_prefix_max_span=1120, mm_prefix_max_ranges=1
    )
    assert (
        len(
            {
                text,
                image,
                tuning._key(
                    *args,
                    mm_prefix_mode="unclamped",
                    mm_prefix_max_span=1089,
                    mm_prefix_max_ranges=1,
                ),
                tuning._key(
                    *args,
                    mm_prefix_mode="clamped",
                    mm_prefix_max_span=560,
                    mm_prefix_max_ranges=1,
                ),
                tuning._key(
                    *args,
                    mm_prefix_mode="clamped",
                    mm_prefix_max_span=1089,
                    mm_prefix_max_ranges=2,
                ),
            }
        )
        == 5
    )


def test_mm_startup_uses_image_sized_query_ceilings_and_skips_clamped_q1():
    """A 1089-token image should not borrow a mostly-text 2048-query profile."""
    workloads = list(
        tuning._workloads(
            4096,
            4096,
            4,
            mm_prefix_mode="clamped",
            mm_prefix_max_span=1152,
            mm_prefix_max_ranges=2,
        )
    )
    queries = {q for _, q, _ in workloads}
    assert 1184 in queries and 2344 in queries
    assert all(q > 1 for q in queries)
    assert not list(
        tuning._workloads(1, 128, 1, max_query_len=1, mm_prefix_mode="clamped")
    )


@pytest.mark.parametrize(
    "dim,logical_page,physical_page", [(512, 16, 64), (256, 64, 16)]
)
def test_layer_warmup_uses_allocated_hybrid_cache_pages(
    monkeypatch, dim, logical_page, physical_page
):
    """A recomputed logical spec must not make warmed profiles miss at runtime."""
    clamps = dim == 256
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
    )
    impl = SimpleNamespace(
        _segmented_attention_config=config,
        sliding_window=(1023, 0) if clamps else (-1, -1),
        num_heads=16,
        num_kv_heads=8 if clamps else 1,
        head_size=dim,
        scale=1.0,
        sinks=None,
    )
    impl._split_kv_cache = lambda cache: (cache[:, 0], cache[:, 1])
    layer = SimpleNamespace(
        impl=impl,
        kv_cache=torch.empty((1, 2, physical_page, dim)),
        get_kv_cache_spec=lambda _: SimpleNamespace(
            block_size=logical_page, dtype=torch.bfloat16
        ),
        mm_prefix_clamp_sliding_window=clamps,
    )
    calls = []
    monkeypatch.setattr(
        tuning, "warmup_segmented_attention", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr(tuning, "_mm_warmup_profiles", lambda *_: [(1152, 1)])
    tuning._warmup_segmented_attention(layer, "cpu", model_clamps_mm_prefix=True)
    assert len(calls) == (2 if clamps else 1)
    assert all(args[5] == physical_page for args in calls)
    assert impl._segmented_attention_warmed_up


def test_mm_specific_winner_survives_persistent_cache_reload(tmp_path):
    """A valid image-only geometry must not be rejected by text-only validation."""
    options = dict(
        mm_prefix_mode="clamped", mm_prefix_max_span=1152, mm_prefix_max_ranges=1
    )
    default = selection.select_segmented_config(
        1, 1117, 1117, 16, 8, 256, True, sliding_window=1023, **options
    )
    configs = tuning._candidate_configs(
        default,
        1,
        1117,
        16,
        256,
        1117,
        native_fp8=True,
        sliding_window=1023,
        mm_prefix_mode="clamped",
    )
    winner = next(
        c
        for c in configs
        if c["bm"] == 64
        and c["bn"] == 64
        and c["warps"] == 8
        and c["stages"] == 2
        and c["kv_upcast"] == "native4"
    )
    record = dict(workload=[1, 1117, 1117], best=winner)
    assert not tuning._valid_record(
        record, 16, 8, 256, True, native_fp8=True, sliding_window=1023
    )
    assert tuning._valid_record(
        record, 16, 8, 256, True, native_fp8=True, sliding_window=1023, **options
    )
    identity = dict(
        kv_upcast_modes=list(selection.KV_UPCAST_MODES), sliding_window=1023, **options
    )
    path = tmp_path / "mm.json"
    path.write_text(json.dumps(dict(identity=identity, records=[record])))
    loaded, _ = tuning._load_records(path, identity, 16, 8, 256, True)
    assert loaded["records"] == [record]


@pytest.mark.parametrize("mode", ["clamped", "unclamped"])
@pytest.mark.parametrize("fp8,query,seq", [(False, 70, 80), (True, 33, 48)])
def test_mm_ceiling_does_not_override_a_different_static_tiling_regime(
    monkeypatch, mode, fp8, query, seq
):
    """A larger query ceiling must not replace a validated short-image tile."""
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    args = ("cuda:0", torch.bfloat16, dtype, 16, 8, 256, 32, 1.0)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode=mode,
        mm_prefix_max_span=64,
        mm_prefix_max_ranges=1,
    )
    best = tuning._normalized_config(
        selection.select_segmented_config(
            1, 128, 128, 16, 8, 256, fp8, native_fp8=fp8, **options
        ),
        1,
        128,
        16,
        256,
    )
    key = tuning._key(*args, **options)
    monkeypatch.setattr(
        tuning,
        "_TABLES",
        {key: {"records": [{"workload": [1, 128, 128], "best": best}]}},
    )
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *_: fp8)
    assert tuning.get_segmented_config(*args, 1, query, seq, **options) is None
    assert tuning.get_segmented_config(*args, 1, 128, 128, **options) == best
    assert tuning.get_segmented_config(*args, 1, 119, 128, **options) == best


@pytest.mark.parametrize(
    "batch,query,seq,tokens,warm_queries,expected_hit",
    [
        (1, 257, 300, 257, [512], False),
        (4, 128, 2048, 131, [128, 128, 128, 128], False),
        (4, 128, 2048, 131, [128, 1, 1, 1], True),
        (1, 512, 512, 512, [512], True),
    ],
)
def test_mm_wider_tuned_tile_preserves_useful_parallelism(
    monkeypatch, batch, query, seq, tokens, warm_queries, expected_hit
):
    """Dense ceilings must not underfill a smaller or newly ragged request."""
    warm_query = max(warm_queries)
    workload = [len(warm_queries), warm_query, 512 if batch == 1 else 2048]
    best = tuning._normalized_config(
        dict(bm=128, bn=32, bk=256, splits=1, warps=8, stages=1, kv_upcast="normal"),
        workload[0],
        workload[1],
        16,
        256,
    )
    args = ("cuda:0", torch.bfloat16, torch.bfloat16, 16, 8, 256, 16, 1.0)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=280,
        mm_prefix_max_ranges=1,
    )
    key = tuning._key(*args, **options)
    monkeypatch.setattr(
        tuning,
        "_TABLES",
        {
            key: {
                "records": [
                    {
                        "workload": workload,
                        "query_lengths": warm_queries,
                        "best": best,
                    }
                ]
            }
        },
    )
    config = tuning.get_segmented_config(
        *args, batch, query, seq, num_query_tokens=tokens, **options
    )
    assert (config is not None) == expected_hit


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("dim", [64, 128, 256, 512])
def test_mm_search_keeps_incumbent_and_all_eligible_conversions(native, dim):
    """Image tile pruning retains a measured fallback and conversion choices."""
    default = selection.select_segmented_config(
        1,
        1024,
        2048,
        16,
        8,
        dim,
        native,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=1089,
    )
    configs = tuning._candidate_configs(
        default, 1, 1024, 16, dim, 2048, native_fp8=native, mm_prefix_mode="clamped"
    )
    incumbent = tuning._normalized_config(default, 1, 1024, 16, dim)
    assert all(configs[0][name] == value for name, value in incumbent.items())
    assert len(configs) == len({tuning._config_key(c) for c in configs})
    geometries = {tuning._geometry_key(c) for c in configs}
    expected = set(selection.KV_UPCAST_MODES) if native else {"normal"}
    assert all(
        {c["kv_upcast"] for c in configs if tuning._geometry_key(c) == geometry}
        == expected
        for geometry in geometries
    )


@pytest.mark.parametrize("dim", [64, 128, 256, 512])
@pytest.mark.parametrize("gqa", [1, 2, 8, 16])
@pytest.mark.parametrize("kv_heads", [1, 3])
@pytest.mark.parametrize("fp8", [False, True])
def test_mm_workspace_reserves_reachable_static_and_tuned_splits(
    dim, gqa, kv_heads, fp8
):
    """An image-specific tile must not outgrow scratch reserved before capture."""
    heads = gqa * kv_heads
    reserved, _ = segmented.segmented_attention_workspace_size(
        4,
        heads,
        kv_heads,
        dim,
        32768,
        max_tokens=2048,
        fp8=fp8,
        autotune=True,
        mm_prefix=True,
        sliding_window=1023,
    )
    for batch, query, image_span, mode in itertools.product(
        (1, 3), (17, 129, 257, 1025), (70, 1089), ("clamped", "unclamped")
    ):
        config = selection.select_segmented_config(
            batch,
            query,
            32768,
            heads,
            kv_heads,
            dim,
            fp8,
            sliding_window=1023,
            mm_prefix_mode=mode,
            mm_prefix_max_span=image_span,
            mm_prefix_max_ranges=1,
        )
        candidates = tuning._candidate_configs(
            config,
            batch,
            query,
            heads,
            dim,
            32768,
            native_fp8=fp8,
            mm_prefix_mode=mode,
            legacy_default=segmented._select_segmented_launch_config(
                batch, query, 32768, heads, kv_heads, dim, fp8
            ),
        )
        splits = max(c["splits"] for c in candidates)
        if splits > 1:
            needed = (
                batch * selection.segmented_query_capacity(query) * heads * dim * splits
            )
            assert needed <= reserved, (
                batch,
                query,
                image_span,
                mode,
                config,
                needed,
                reserved,
            )


@pytest.mark.parametrize("dim", (64, 128, 256, 512))
@pytest.mark.parametrize("gqa", range(1, 17))
@pytest.mark.parametrize("query_len,seq_len", [(1, 32768), (17, 1024), (257, 32768)])
def test_every_eligible_launch_searches_all_conversion_modes(
    dim, gqa, query_len, seq_len
):
    """Changing geometry must not silently drop normal or either native mode."""
    heads, kv_heads = 2 * gqa, 2
    default = segmented.select_segmented_config(
        1, query_len, seq_len, heads, kv_heads, dim, True
    )
    legacy = segmented._select_segmented_launch_config(
        1, query_len, seq_len, heads, kv_heads, dim, True
    )
    configs = tuning._candidate_configs(
        default,
        1,
        query_len,
        heads,
        dim,
        seq_len,
        native_fp8=True,
        legacy_default=legacy,
    )
    assert configs[0] == tuning._normalized_config(default, 1, query_len, heads, dim)
    assert len(configs) == len({tuning._config_key(c) for c in configs})
    launches: dict[tuple, set[str]] = {}
    for config in configs:
        geometry = tuple(sorted((k, v) for k, v in config.items() if k != "kv_upcast"))
        launches.setdefault(geometry, set()).add(config["kv_upcast"])
    assert all(modes == set(segmented.KV_UPCAST_MODES) for modes in launches.values())
    initial = tuning._initial_search_configs(configs)
    assert initial[0] == configs[0]
    assert {tuning._geometry_key(c) for c in initial} == set(launches)
    assert all(c == configs[0] or c["kv_upcast"] == "native4" for c in initial)
    assert len(initial) == len({tuning._config_key(c) for c in initial})
    incumbent_launches = tuning._launch_candidate_configs(
        legacy, 1, query_len, heads, dim, seq_len
    )
    assert all(
        tuple(sorted((k, v) for k, v in c.items() if k != "kv_upcast")) in launches
        for c in incumbent_launches
    )
    limit = min(64, 2 * default["splits"]) if dim == 64 else default["splits"]
    assert all(c["splits"] <= limit for c in configs)


@pytest.mark.parametrize("dim", (64, 128, 256, 512))
def test_ineligible_search_deduplicates_inactive_native_modes(dim):
    """BF16 caches, FP16 queries, and unsupported hardware need one conversion."""
    default = segmented.select_segmented_config(1, 1, 8192, 16, 2, dim, True)
    configs = tuning._candidate_configs(default, 1, 1, 16, dim, native_fp8=False)
    assert all(c["kv_upcast"] == "normal" for c in configs)
    assert len(configs) == len({tuning._config_key(c) for c in configs})
    assert tuning._initial_search_configs(configs) == configs
    assert tuning._upcast_expansion_configs(configs, []) == []


def _shortlist_configs():
    return [
        dict(
            bm=bm,
            bn=bn,
            bk=64,
            splits=1,
            warps=warps,
            stages=1,
            waves_per_eu=2,
            reduce_d=64,
            reduce_warps=4,
            kv_upcast=mode,
        )
        for bm in (16, 32, 64)
        for bn in (32, 64)
        for warps in (4, 8)
        for mode in segmented.KV_UPCAST_MODES
    ]


@pytest.mark.parametrize("failed_anchor", (False, True))
def test_conversion_shortlist_keeps_default_and_recovers_failed_anchors(failed_anchor):
    """An unmeasurable native4 tile must not hide its other conversions."""
    configs = _shortlist_configs()
    anchors = [c for c in configs if c["kv_upcast"] == "native4"]
    comparisons = [
        dict(config=c, paired_balanced_speedup_vs_default=1 + index / 100)
        for index, c in enumerate(anchors)
        if not (failed_anchor and index == 1)
    ]
    expanded = tuning._upcast_expansion_configs(configs, comparisons)
    expected = {tuning._geometry_key(c) for c in anchors[-8:]}
    expected.add(tuning._geometry_key(configs[0]))
    if failed_anchor:
        expected.add(tuning._geometry_key(anchors[1]))
    assert {tuning._geometry_key(c) for c in expanded} == expected
    initial_keys = {
        tuning._config_key(c) for c in tuning._initial_search_configs(configs)
    }
    assert not initial_keys.intersection(tuning._config_key(c) for c in expanded)
    combined = tuning._initial_search_configs(configs) + expanded
    for geometry in expected:
        assert {
            c["kv_upcast"] for c in combined if tuning._geometry_key(c) == geometry
        } == set(segmented.KV_UPCAST_MODES)


@pytest.mark.parametrize("mm_mode", ["none", "clamped", "unclamped"])
def test_staged_tuning_uses_contemporaneous_baselines_and_compiles_only_shortlist(
    monkeypatch,
    mm_mode,
):
    """Clock drift between stages must not suppress a better native1 candidate."""
    configs = _shortlist_configs()
    geometries = list(dict.fromkeys(tuning._geometry_key(c) for c in configs))
    compiled = []
    measure_calls = 0

    def measure(candidates, run, device, eviction, rounds, graph_calls):
        nonlocal measure_calls
        measure_calls += 1
        baseline = 200 if measure_calls in (3, 4) else 100
        samples = {}
        for config in candidates:
            index = geometries.index(tuning._geometry_key(config))
            mode = config["kv_upcast"]
            factor = 1.0
            if mode == "native4":
                factor = 0.95 - index * 0.005
            elif mode == "native1":
                factor = 0.8 if index == 8 else 1.08
            samples[tuning._config_key(config)] = [baseline * factor] * rounds
        return samples

    monkeypatch.setattr(tuning, "select_segmented_config", lambda *a, **kw: configs[0])
    monkeypatch.setattr(tuning, "_candidate_configs", lambda *a, **kw: configs)
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *a: True)
    monkeypatch.setattr(
        tuning,
        "_precompile_configs",
        lambda candidates, *a: compiled.append(candidates),
    )
    monkeypatch.setattr(
        tuning, "run_segmented_attention", lambda q, out, *a, **kw: out.copy_(q)
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a: None)
    monkeypatch.setattr(tuning, "_condition_configs", lambda *a: None)
    monkeypatch.setattr(tuning, "_measure_configs", measure)
    record = tuning._tune_workload(
        "cpu",
        torch.bfloat16,
        torch.float8_e4m3fn,
        1,
        1,
        64,
        16,
        0.125,
        16,
        (1, 1, 128),
        mm_prefix_mode=mm_mode,
        mm_prefix_max_span=70 if mm_mode != "none" else 0,
        mm_prefix_max_ranges=2 if mm_mode != "none" else 0,
    )
    assert record["best"]["kv_upcast"] == "native1"
    assert tuning._geometry_key(record["best"]) == geometries[8]
    assert len(compiled) == 2
    assert sum(map(len, compiled)) == len(record["results"]) < len(configs)
    assert len({tuning._config_key(c) for batch in compiled for c in batch}) == sum(
        map(len, compiled)
    )
    assert record["search"]["screen_baselines"][0]["cold_us"] == [100] * 5
    assert record["search"]["screen_baselines"][1]["cold_us"] == [200] * 5


@pytest.mark.parametrize("gfx12", (False, True))
@pytest.mark.parametrize(
    "dtype,kv_dtype,eligible",
    [
        (torch.bfloat16, torch.float8_e4m3fn, True),
        (torch.bfloat16, torch.float8_e4m3fnuz, False),
        (torch.float16, torch.float8_e4m3fn, False),
        (torch.bfloat16, torch.bfloat16, False),
    ],
)
def test_native_gate_checks_query_format_cache_format_and_architecture(
    monkeypatch, gfx12, dtype, kv_dtype, eligible
):
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(on_gfx12x=lambda: gfx12),
    )
    assert segmented.supports_segmented_native_fp8(dtype, kv_dtype) == (
        gfx12 and eligible
    )


def test_normal_mode_overrides_legacy_native_flags():
    """An explicit normal choice must survive configs built from old templates."""
    assert (
        segmented.segmented_kv_upcast_mode(
            {"kv_upcast": "normal", "native_fp8_upcast": True, "native_fp8_pack": 4}
        )
        == "normal"
    )
    assert (
        segmented.segmented_kv_upcast_mode(
            {"native_fp8_upcast": True, "native_fp8_pack": 4}
        )
        == "native4"
    )


@pytest.mark.parametrize(
    "warm,cold,promote", [(80, 100, True), (60, 104, False), (99, 100, False)]
)
def test_balanced_promotion_realizes_warm_gains_without_cold_regressions(
    warm, cold, promote
):
    incumbent = {"kv_upcast": "normal"}
    challenger = {"kv_upcast": "native4"}
    keys = list(map(tuning._config_key, (incumbent, challenger)))
    winner, _ = tuning._select_tuned_winner(
        incumbent,
        [incumbent, challenger],
        {keys[0]: [100.0] * 7, keys[1]: [float(cold)] * 7},
        warm_samples={keys[0]: [100.0] * 7, keys[1]: [float(warm)] * 7},
    )
    assert winner["config"] == (challenger if promote else incumbent)


def test_cache_record_validation_respects_native_eligibility():
    default = segmented.select_segmented_config(1, 1, 128, 8, 2, 128, True)
    config = dict(tuning._normalized_config(default, 1, 1, 8, 128), kv_upcast="native4")
    record = {"workload": [1, 1, 128], "best": config}
    assert tuning._valid_record(record, 8, 2, 128, True, native_fp8=True)
    assert not tuning._valid_record(record, 8, 2, 128, True, native_fp8=False)


def test_cache_identity_invalidates_records_after_selector_changes(
    monkeypatch, tmp_path
):
    """The extracted selector must remain part of persistent cache identity."""
    selector = tmp_path / "selector.py"
    selector.write_text("initial selection")
    monkeypatch.setattr(tuning.segmented_attention_config, "__file__", str(selector))
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *a: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="test GPU", gcnArchName="gfx1201", multi_processor_count=64
        ),
    )
    args = (
        "cuda:0",
        torch.bfloat16,
        torch.float8_e4m3fn,
        8,
        2,
        128,
        32,
        0.125,
        8192,
        131072,
        32,
        -1,
        True,
        8192,
    )
    first = tuning._identity(*args)
    selector.write_text("updated selection")
    second = tuning._identity(*args)
    assert first["selector_sha256"] != second["selector_sha256"]
    assert {k: v for k, v in first.items() if k != "selector_sha256"} == {
        k: v for k, v in second.items() if k != "selector_sha256"
    }


@pytest.mark.parametrize("dim", (64, 128, 256, 512))
@pytest.mark.parametrize("mode", segmented.KV_UPCAST_MODES)
@pytest.mark.parametrize("supported", (False, True))
@pytest.mark.parametrize("compile_only", (False, True))
def test_explicit_conversion_reaches_runtime_and_compile_specializations(
    monkeypatch, dim, mode, supported, compile_only
):
    """Both launch paths honor the selected mode and the hardware gate."""
    calls = []

    class Stage:
        def __getitem__(self, grid):
            return self.warmup

        def warmup(self, *args, **kwargs):
            calls.append(args)

    monkeypatch.setattr(segmented, "_segmented_attention_stage", Stage())
    monkeypatch.setattr(
        segmented, "supports_segmented_native_fp8", lambda *_: supported
    )
    q = torch.empty((1, 4, dim), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    cache = torch.empty((1, 16, 2, dim), dtype=torch.float8_e4m3fn)
    table = torch.zeros((1, 1), dtype=torch.int32)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    scale = torch.tensor(1.0)
    config = dict(bm=16, bn=32, bk=64, splits=1, warps=4, stages=1, kv_upcast=mode)
    args = (q, output, cache, cache, table, starts, lengths, 1)
    if compile_only:
        segmented.compile_segmented_attention(*args, scale, scale, 1.0, config, None)
    else:
        assert (
            segmented.run_segmented_attention(
                *args, 1, scale, scale, 1.0, config=config
            )
            is output
        )
    assert len(calls) == 1
    native = supported and mode != "normal"
    assert calls[0][-2:] == (native, 4 if native and mode == "native4" else 1)


@pytest.mark.parametrize("query_len,splits", [(17, 1), (65, 1), (17, 2), (65, 2)])
def test_single_pass_compile_reuses_capacity_without_shrinking_the_grid(
    monkeypatch, query_len, splits
):
    """Unused scratch indexing must not compile a new single-pass stage per Q."""
    position = segmented._segmented_attention_stage.arg_names.index("QCAP")
    calls = []

    class Kernel:
        def warmup(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(segmented, "_segmented_attention_stage", Kernel())
    monkeypatch.setattr(segmented, "_segmented_attention_reduce", Kernel())
    q = torch.empty((query_len, 2, 64), dtype=torch.bfloat16)
    cache = torch.empty((8, 32, 1, 64), dtype=torch.bfloat16)
    table = torch.zeros((1, 8), dtype=torch.int32)
    starts = torch.tensor([0, query_len], dtype=torch.int32)
    lens = torch.tensor([256], dtype=torch.int32)
    scale = torch.tensor(1.0)
    qcap = selection.segmented_query_capacity(query_len)
    shapes = segmented.segmented_workspace_shapes(1, qcap, 2, 1, 64, splits)
    workspace = None if shapes is None else tuple(torch.empty(s) for s in shapes)
    cfg = dict(bm=16, bn=32, bk=64, splits=splits, warps=4, stages=1)
    segmented.compile_segmented_attention(
        q,
        torch.empty_like(q),
        cache,
        cache,
        table,
        starts,
        lens,
        query_len,
        scale,
        scale,
        1.0,
        cfg,
        workspace,
    )
    args, options = calls[0]
    assert args[position] == (qcap if splits > 1 else 1)
    assert options["grid"] == (1, (qcap * 2 + 15) // 16, splits)


@pytest.mark.parametrize("graph_calls", (0, 1, 32))
def test_graph_timing_excludes_eviction_and_normalizes_call_count(
    monkeypatch, graph_calls
):
    """The startup objective must measure kernels independently of eviction cost."""
    clock = SimpleNamespace(us=0.0, capture=None, evictions=0, order=[], captures=[])

    def emit(operation):
        if clock.capture is None:
            operation()
        else:
            clock.capture.append(operation)

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True, "external": True}
            self.time = 0.0

        def record(self):
            emit(lambda: setattr(self, "time", clock.us))

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return (end.time - self.time) / 1000

    class Graph:
        def __init__(self):
            self.operations: list = []

        def replay(self):
            for operation in self.operations:
                operation()

    @contextmanager
    def capture(graph):
        clock.capture = graph.operations
        yield
        clock.capture = None

    def evict():
        def operation():
            clock.us += 1000
            clock.evictions += 1

        emit(operation)

    def run(config):
        clock.captures.append(config["cost_us"])

        def operation():
            clock.us += config["cost_us"]
            clock.order.append(config["cost_us"])

        emit(operation)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda _: None)
    configs = [{"cost_us": 10.0}, {"cost_us": 20.0}]
    samples = tuning._measure_configs(
        configs, run, "cuda:0", SimpleNamespace(zero_=evict), 5, graph_calls
    )
    assert samples == {tuning._config_key(c): [c["cost_us"]] * 5 for c in configs}
    assert clock.evictions == (12 if graph_calls <= 1 else 0)
    calls = max(1, graph_calls)
    assert clock.captures[::calls] == [10.0, 20.0]
    first_in_round = clock.order[2 * calls :: 2 * calls]
    assert abs(first_in_round.count(10.0) - first_in_round.count(20.0)) <= 1


@pytest.mark.parametrize("dim", (128, 256, 512))
def test_ceiling_lookup_preserves_parallelism_within_reserved_workspace(dim):
    """Q259 needs more splits than the Q512 ceiling that supplied its tile."""
    best = tuning._normalized_config(
        segmented.select_segmented_config(1, 512, 32768, 1, 1, dim, True),
        1,
        512,
        1,
        dim,
    )
    best["splits"] = 16
    record = {"workload": [1, 512, 32768], "best": best}
    adapted = tuning._runtime_config(record, 1, 259, 16411, 1, 1, dim, True)
    assert best["splits"] == 16 and adapted["splits"] == 32
    assert record["best"] == best
    assert tuning._runtime_config(record, 1, 512, 32768, 1, 1, dim, True) == best
    sizes = segmented.segmented_attention_workspace_size(
        1, 1, 1, dim, 32768, max_tokens=512, fp8=True, autotune=True
    )
    shapes = segmented.segmented_workspace_shapes(1, 512, 1, 1, dim, adapted["splits"])
    assert all(torch.Size(s).numel() <= limit for s, limit in zip(shapes, sizes))


def test_lookup_keeps_verified_long_decode_splits_across_sequence_boundary():
    """A one-token sequence boundary must not discard the tuned 512-way decode."""
    best = dict(bm=16, splits=512, kv_upcast="native4")
    record = {"workload": [1, 1, 131072], "best": best}
    assert tuning._runtime_config(record, 1, 1, 131071, 1, 1, 64, True) == best


@pytest.mark.parametrize("dim", (128, 256, 512))
@pytest.mark.parametrize("gqa", (1, 3, 8, 16))
def test_narrow_window_lookup_keeps_a_single_pass_winner(monkeypatch, dim, gqa):
    """Short windows must not pay for a reduction absent from startup tuning."""
    best = tuning._normalized_config(
        segmented.select_segmented_config(1, 512, 512, gqa, 1, dim, True),
        1,
        512,
        gqa,
        dim,
    )
    best["splits"] = 1
    record = {"workload": [1, 512, 512], "best": best}
    key = tuning._key(
        "cuda:0", torch.bfloat16, torch.float8_e4m3fn, gqa, 1, dim, 32, 1.0, 127
    )
    monkeypatch.setattr(tuning, "_TABLES", {key: {"records": [record]}})
    actual = tuning.get_segmented_config(
        "cuda:0",
        torch.bfloat16,
        torch.float8_e4m3fn,
        gqa,
        1,
        dim,
        32,
        1.0,
        1,
        259,
        386,
        sliding_window=127,
    )
    assert actual == best
    assert record["best"] == best


def test_wider_window_lookup_can_add_useful_partition_parallelism():
    """A ragged batch below its warm ceiling still benefits from more CTAs."""
    best = tuning._normalized_config(
        segmented.select_segmented_config(4, 32, 2048, 14, 1, 128, True),
        4,
        32,
        14,
        128,
    )
    best["splits"] = 1
    record = {"workload": [4, 32, 2048], "best": best}
    actual = tuning._runtime_config(
        record, 3, 19, 1042, 14, 1, 128, True, sliding_window=1023
    )
    assert actual["splits"] > 1 and record["best"]["splits"] == 1


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mm_shuffled_tables_cover_disjoint_cache_pages_and_preserve_padding(kv_dtype):
    inputs = tuning._make_inputs(
        "cpu",
        torch.bfloat16,
        kv_dtype,
        2,
        1,
        64,
        32,
        64,
        (3, 16, 128),
        table_max_len=1024,
        shuffle_pages=True,
    )
    table = inputs[5]
    pages = table[:, :4]
    assert table.shape == (3, 32) and table.dtype == torch.int32
    assert sorted(pages.flatten().tolist()) == list(range(12))
    assert torch.equal(table, pages.repeat(1, 8))
    assert inputs[7].tolist() == [128, 128, 128]


@pytest.mark.parametrize("matched_density", [False, True])
def test_mm_lookup_rejects_only_unmeasured_sparse_query_density(
    monkeypatch, matched_density
):
    query_lengths = [256, 5, 1, 1, 1, 1, 1, 0]
    tokens = sum(query_lengths)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=1089,
        mm_prefix_max_ranges=1,
    )
    args = ("cuda:0", torch.bfloat16, torch.float8_e4m3fn, 16, 8, 256, 16, 1.0)
    best = tuning._normalized_config(
        selection.select_segmented_config(8, 256, 8192, 16, 8, 256, True, **options),
        8,
        256,
        16,
        256,
    )
    best.update({"bn": 32, "warps": 4})
    record = dict(
        workload=[8, 256, 8192],
        best=best,
        query_lengths=query_lengths if matched_density else [256] * 8,
    )
    monkeypatch.setattr(
        tuning, "_TABLES", {tuning._key(*args, **options): {"records": [record]}}
    )
    monkeypatch.setattr(tuning, "supports_segmented_native_fp8", lambda *_: True)
    result = tuning.get_segmented_config(
        *args, 8, 256, 2367, num_query_tokens=tokens, **options
    )
    assert (result is not None) == matched_density


def test_mm_grid_retains_short_profiles_midpoints_and_two_request_batches():
    workloads = list(
        tuning._workloads(
            4096,
            4096,
            4,
            mm_prefix_mode="clamped",
            mm_prefix_max_span=1152,
            mm_prefix_max_ranges=2,
        )
    )
    assert {4, 16, 32, 48, 64, 96, 256, 384, 1536}.issubset(
        {q for _, q, _ in workloads}
    )
    assert any(b == 2 and q == 384 for b, q, _ in workloads)

    assert (1, 48, 64) in workloads


@pytest.mark.parametrize("fp8,partitions", [(False, {1, 2, 4}), (True, {1, 4, 8})])
def test_mm_search_preserves_incumbent_partition_alternatives(fp8, partitions):
    default = selection.select_segmented_config(
        1,
        64,
        8192,
        16,
        8,
        256,
        fp8,
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=1152,
        mm_prefix_max_ranges=1,
    )
    configs = tuning._candidate_configs(
        default,
        1,
        64,
        16,
        256,
        8192,
        native_fp8=fp8,
        mm_prefix_mode="clamped",
        sliding_window=1023,
    )
    assert partitions.issubset(
        {
            c["splits"]
            for c in configs
            if c["bm"] == default["bm"]
            and c["bn"] == default["bn"]
            and c["bk"] == default["bk"]
            and c["warps"] == default["warps"]
            and c["stages"] == default["stages"]
        }
    )


@pytest.mark.parametrize(
    "seq_len,expected_splits", [(32, 1), (64, 1), (65, 2), (128, 2)]
)
def test_mm_runtime_discards_partitions_without_cache_tiles(seq_len, expected_splits):
    best = tuning._normalized_config(
        selection.select_segmented_config(1, 32, 8192, 16, 8, 256, True),
        1,
        32,
        16,
        256,
    )
    best.update({"bn": 64, "splits": 8})
    record = {"workload": [1, 32, 2048], "best": best}
    result = tuning._runtime_config(
        record,
        1,
        32,
        seq_len,
        16,
        8,
        256,
        True,
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=64,
        mm_prefix_max_ranges=1,
        native_fp8=True,
    )
    assert result["splits"] == expected_splits
    assert record["best"]["splits"] == 8


@pytest.mark.parametrize(
    "query,seq_len,bm,bn,expected_hit",
    [(128, 128, 64, 64, False), (128, 128, 64, 128, True), (64, 64, 32, 64, True)],
)
def test_exact_short_mm_lookup_keeps_kv_reuse_and_measured_wider_tiles(
    monkeypatch, query, seq_len, bm, bn, expected_hit
):
    args = ("cuda:0", torch.bfloat16, torch.bfloat16, 16, 8, 256, 16, 1.0)
    options = dict(
        sliding_window=1023,
        mm_prefix_mode="clamped",
        mm_prefix_max_span=query - 12,
        mm_prefix_max_ranges=1,
    )
    best = tuning._normalized_config(
        selection.select_segmented_config(
            1, query, seq_len, 16, 8, 256, False, **options
        ),
        1,
        query,
        16,
        256,
    )
    best.update({"bm": bm, "bn": bn, "warps": 8, "stages": 1, "splits": 1})
    record = {"workload": [1, query, seq_len], "best": best, "query_lengths": [query]}
    monkeypatch.setattr(
        tuning, "_TABLES", {tuning._key(*args, **options): {"records": [record]}}
    )
    result = tuning.get_segmented_config(
        *args, 1, query, seq_len, num_query_tokens=query, **options
    )
    assert (result is not None) == expected_hit
    assert record["best"]["bm"] == bm
