# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Direct-finalize RDNA4 SplitKV stage."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from .rdna4_splitkv_common import (
    BLOCK_THREADS,
    LOG2E,
    PARTITIONS,
    TILE_TOKENS,
    WAVE_SIZE,
    _dequant_fp8x8,
    _flat_view,
    _wave_reduce,
)


@functools.lru_cache(maxsize=128)
def compile_direct_stage(
    *,
    query_dtype: str,
    kv_dtype: str,
    splits: int,
    num_kv_heads: int,
    query_group_size: int,
    head_dim: int,
    page_size: int,
    softmax_scale: float,
):
    """Build one eight-wave grouped-query stage-1 launcher."""

    if query_dtype not in ("bf16", "fp16"):
        raise ValueError(f"query_dtype must be 'bf16' or 'fp16', got {query_dtype!r}")
    if kv_dtype not in ("bf16", "fp8", "fp8fnuz"):
        raise ValueError(
            f"kv_dtype must be 'bf16', 'fp8', or 'fp8fnuz', got {kv_dtype!r}"
        )
    if num_kv_heads < 1:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if not 1 <= query_group_size <= 4:
        raise ValueError(
            f"query_group_size must be between 1 and 4, got {query_group_size}"
        )
    if head_dim not in (128, 256):
        raise ValueError(f"head_dim must be 128 or 256, got {head_dim}")
    if page_size < 8 or page_size % 8:
        raise ValueError(f"page_size must be a positive multiple of 8, got {page_size}")
    is_fp8 = kv_dtype in ("fp8", "fp8fnuz")
    is_fp8fnuz = kv_dtype == "fp8fnuz"
    is_fp16_query = query_dtype == "fp16"

    k_bytes = TILE_TOKENS * head_dim * 2
    score_offset = k_bytes
    score_bytes = 16 * TILE_TOKENS * 4
    v_bytes = head_dim * TILE_TOKENS * 2
    prob_offset = v_bytes
    prob_bytes = 16 * TILE_TOKENS * 2
    alpha_offset = prob_offset + prob_bytes
    alpha_bytes = PARTITIONS * 4 * 4
    q_offset = max(score_offset + score_bytes, alpha_offset + alpha_bytes) + 2048
    base_lds_bytes = q_offset + 16 * head_dim * 2
    partial_offset = base_lds_bytes
    partial_bytes = PARTITIONS * 4 * head_dim * 4
    max_offset = partial_offset + partial_bytes
    max_bytes = PARTITIONS * 4 * 4
    sum_offset = max_offset + max_bytes
    total_lds_bytes = sum_offset + PARTITIONS * 4 * 4

    @fx.struct
    class SharedStorage:
        words: fx.Array[fx.Int32, total_lds_bytes // 4, 16]

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def grouped_stage1(
        query_ptr: fx.Tensor,
        key_ptr: fx.Tensor,
        value_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        seq_lens_ptr: fx.Tensor,
        query_start_loc_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        v_scale_ptr: fx.Tensor,
        mid_out_ptr: fx.Tensor,
        mid_lse_ptr: fx.Tensor,
        batch: fx.Int32,
        table_stride: fx.Int32,
        q_stride0: fx.Int32,
        q_stride1: fx.Int32,
        k_stride0: fx.Int32,
        k_stride1: fx.Int32,
        k_stride2: fx.Int32,
        k_stride3: fx.Int32,
        k_stride4: fx.Int32,
        v_stride0: fx.Int32,
        v_stride1: fx.Int32,
        v_stride2: fx.Int32,
        v_stride3: fx.Int32,
        mo_stride0: fx.Int32,
        mo_stride1: fx.Int32,
        mo_stride2: fx.Int32,
        ml_stride0: fx.Int32,
        ml_stride1: fx.Int32,
        ml_stride2: fx.Int32,
        qk_mma: fx.TiledMma,
        pv_mma: fx.TiledMma,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        block = fx.Int32(gpu.block_id("x"))
        split = block % splits
        item = block // splits
        seq = item // num_kv_heads
        kv_head = item % num_kv_heads

        query = _flat_view(query_ptr)
        key = _flat_view(key_ptr)
        value = _flat_view(value_ptr)
        block_tables = _flat_view(block_tables_ptr)
        seq_lens = _flat_view(seq_lens_ptr)
        query_start_loc = _flat_view(query_start_loc_ptr)
        k_scales = _flat_view(k_scale_ptr)
        v_scales = _flat_view(v_scale_ptr)
        mid_out = _flat_view(mid_out_ptr)

        def make_64b_prefetcher(tensor_ptr, elem_type, width, slots, *, recast=False):
            atom = fx.make_copy_atom(fx.UniversalCopy64b(), elem_type)
            registers = [
                fx.make_rmem_tensor(fx.make_layout(width, 1), elem_type)
                for _ in range_constexpr(slots)
            ]
            base_iter = fx.get_iter(tensor_ptr)
            if recast:
                base_iter = fx.recast_iter(elem_type, base_iter)
            flat = fx.make_view(base_iter, fx.make_layout(1 << 30, 1))
            divided = fx.logical_divide(flat, fx.make_layout(1, 1))

            def prefetch(slot, element_offset):
                fx.copy(
                    atom,
                    fx.slice(divided, (None, element_offset)),
                    registers[slot],
                )

            def consume(slot):
                return fx.Vector(fx.memref_load_vec(registers[slot]))

            return prefetch, consume

        def make_64b_loader(tensor_ptr, elem_type, width, *, recast=False):
            atom = fx.make_copy_atom(fx.UniversalCopy64b(), elem_type)
            register = fx.make_rmem_tensor(fx.make_layout(width, 1), elem_type)
            base_iter = fx.get_iter(tensor_ptr)
            if recast:
                base_iter = fx.recast_iter(elem_type, base_iter)
            flat = fx.make_view(base_iter, fx.make_layout(1 << 30, 1))
            divided = fx.logical_divide(flat, fx.make_layout(1, 1))

            def load(element_offset):
                fx.copy(
                    atom,
                    fx.slice(divided, (None, element_offset)),
                    register,
                )
                return fx.Vector(fx.memref_load_vec(register))

            return load

        if const_expr(is_fp8):
            load_key_u8x8 = make_64b_loader(key_ptr, fx.Uint8, 8, recast=True)
            prefetch_value_u8x8, consume_value_u8x8 = make_64b_prefetcher(
                value_ptr, fx.Uint8, 8, 8, recast=True
            )

        def dequant_fp8x8(raw, scale):
            return _dequant_fp8x8(raw, scale, fx.BFloat16, is_fp8fnuz=is_fp8fnuz)

        k_scale = fx.Float32(1.0)
        v_scale = fx.Float32(1.0)
        if const_expr(is_fp8):
            k_scale = fx.Float32(k_scales[0])
            v_scale = fx.Float32(v_scales[0])

        query_row = fx.Int32(query_start_loc[seq])
        is_decode = fx.Int32(query_start_loc[seq + 1]) - query_row == 1
        length = is_decode.select(fx.Int32(seq_lens[seq]), fx.Int32(0))
        tiles = (length + TILE_TOKENS - 1) // TILE_TOKENS
        tiles_per_split = (tiles + splits - 1) // splits
        begin = split * tiles_per_split * TILE_TOKENS
        end = fx.min(begin + tiles_per_split * TILE_TOKENS, length)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        byte_base = fx.recast_iter(fx.Uint8, storage.words.ptr)

        def shared_view(byte_offset, elem_type, shape, stride):
            ptr = fx.add_offset(byte_base, fx.Int32(byte_offset))
            return fx.make_view(
                fx.recast_iter(elem_type, ptr), fx.make_layout(shape, stride)
            )

        def lds_barrier():
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        s_query = shared_view(q_offset, fx.BFloat16, (16, head_dim), (head_dim, 1))
        s_key = shared_view(0, fx.BFloat16, (TILE_TOKENS, head_dim), (head_dim, 1))
        s_score = shared_view(
            score_offset, fx.Float32, (16, TILE_TOKENS), (TILE_TOKENS, 1)
        )
        s_value = shared_view(0, fx.BFloat16, (head_dim, TILE_TOKENS), (TILE_TOKENS, 1))
        s_prob = shared_view(
            prob_offset, fx.BFloat16, (16, TILE_TOKENS), (TILE_TOKENS, 1)
        )
        s_alpha = shared_view(alpha_offset, fx.Float32, (PARTITIONS, 4), (4, 1))
        s_partial = shared_view(
            partial_offset,
            fx.Float32,
            (PARTITIONS, 4, head_dim),
            (4 * head_dim, head_dim, 1),
        )
        s_max = shared_view(max_offset, fx.Float32, (PARTITIONS, 4), (4, 1))
        s_sum = shared_view(sum_offset, fx.Float32, (PARTITIONS, 4), (4, 1))

        def load_fp8_key_fragment(tile_base, token_wave):
            token = fx.Int32(tile_base) + token_wave * 16 + lane % 16
            valid = token < end
            safe_token = valid.select(token, fx.Int32(0))
            logical_page = safe_token // page_size
            in_page = safe_token - logical_page * page_size
            physical_page = fx.Int32(block_tables[seq * table_stride + logical_page])
            d_half = (lane // 16) * 8
            fragment_values = []
            for d_fragment in range_constexpr(head_dim // 16):
                key_index = (
                    physical_page * k_stride0
                    + kv_head * k_stride1
                    + d_fragment * k_stride2
                    + in_page * k_stride3
                    + d_half * k_stride4
                )
                key_values = dequant_fp8x8(load_key_u8x8(key_index), k_scale)
                for value_index in range_constexpr(8):
                    fragment_values.append(
                        valid.select(key_values[value_index], fx.BFloat16(0.0))
                    )
            fx.rocdl.s_wait_loadcnt(0)
            return fx.Vector.from_elements(fragment_values, dtype=fx.BFloat16)

        # Q is invariant across context tiles. Pad ten inactive query rows with
        # zero so the 16-row WMMA tile remains regular.
        for q_iter in range_constexpr(16):
            linear = tid + q_iter * BLOCK_THREADS
            q_local = linear // head_dim
            d = linear % head_dim
            valid_q = q_local < query_group_size
            safe_q = valid_q.select(q_local, fx.Int32(0))
            qh = kv_head * query_group_size + safe_q
            q_value = fx.Float32(query[query_row * q_stride0 + qh * q_stride1 + d])
            scaled_q = (q_value * softmax_scale).to(fx.BFloat16)
            s_query[q_local, d] = valid_q.select(scaled_q, fx.BFloat16(0.0))
        lds_barrier()

        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        for init_iter in range_constexpr((PARTITIONS * 4 * head_dim) // BLOCK_THREADS):
            linear = tid + init_iter * BLOCK_THREADS
            partition = linear // (4 * head_dim)
            row_d = linear % (4 * head_dim)
            s_partial[partition, row_d // head_dim, row_d % head_dim] = zero
        if tid < PARTITIONS * 4:
            s_max[tid // 4, tid % 4] = neg_inf
            s_sum[tid // 4, tid % 4] = zero
        lds_barrier()

        for tile in range(begin, end, TILE_TOKENS):
            # BF16 uses a shared K tile. FP8 is loaded directly into the native
            # WMMA B fragments below, avoiding a widened BF16 LDS handoff.
            if const_expr(not is_fp8):
                for load_iter in range_constexpr(
                    (TILE_TOKENS * head_dim) // BLOCK_THREADS
                ):
                    linear = tid + load_iter * BLOCK_THREADS
                    token_local = linear // head_dim
                    d = linear % head_dim
                    token = fx.Int32(tile) + token_local
                    valid = token < end
                    safe_token = valid.select(token, fx.Int32(0))
                    logical_page = safe_token // page_size
                    in_page = safe_token - logical_page * page_size
                    physical_page = fx.Int32(
                        block_tables[seq * table_stride + logical_page]
                    )
                    key_index = (
                        physical_page * k_stride0
                        + kv_head * k_stride1
                        + (d // 8) * k_stride2
                        + in_page * k_stride3
                        + (d % 8) * k_stride4
                    )
                    s_key[token_local, d] = valid.select(
                        key[key_index], fx.BFloat16(0.0)
                    )
                lds_barrier()

            # Four waves compute [16,256] @ [64,256]^T -> [16,64].
            if wave < 4:
                key_wave_ptr = fx.add_offset(fx.get_iter(s_key), wave * 16 * head_dim)
                key_wave = fx.make_view(
                    key_wave_ptr,
                    fx.make_layout((16, head_dim), (head_dim, 1)),
                )
                score_wave_ptr = fx.add_offset(fx.get_iter(s_score), wave * 16)
                score_wave = fx.make_view(
                    score_wave_ptr,
                    fx.make_layout((16, 16), (TILE_TOKENS, 1)),
                )
                qk_thread = qk_mma.thr_slice(lane)
                frag_q = qk_thread.make_fragment_A(s_query)
                frag_k = qk_thread.make_fragment_B(key_wave)
                frag_s = qk_thread.make_fragment_C(score_wave)
                copy_bf16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
                copy_q = fx.make_tiled_copy_A(copy_bf16, qk_mma).get_slice(lane)
                fx.copy(
                    copy_bf16,
                    copy_q.partition_S(s_query),
                    copy_q.retile(frag_q),
                )
                if const_expr(is_fp8):
                    frag_k.store(load_fp8_key_fragment(tile, wave))
                else:
                    copy_k = fx.make_tiled_copy_B(copy_bf16, qk_mma).get_slice(lane)
                    fx.copy(
                        copy_bf16,
                        copy_k.partition_S(key_wave),
                        copy_k.retile(frag_k),
                    )
                frag_s.fill(0.0)
                for k_repeat in range_constexpr(head_dim // 16):
                    fx.gemm(
                        qk_mma,
                        frag_s,
                        frag_q[None, None, k_repeat],
                        frag_k[None, None, k_repeat],
                        frag_s,
                    )
                score_values = fx.Vector(frag_s.load())
                score_row_base = (lane // 16) * 8
                score_col = wave * 16 + lane % 16
                for score_row in range_constexpr(8):
                    s_score[score_row_base + score_row, score_col] = score_values[
                        score_row
                    ]

            # The gfx120x repeated-K path drops accumulator row 4 in the
            # lower producer-wave branch. The otherwise-idle upper waves
            # recompute one 16-token slice each and publish only that row.
            if wave >= 4:
                repair_wave = wave - 4
                repair_key_ptr = fx.add_offset(
                    fx.get_iter(s_key), repair_wave * 16 * head_dim
                )
                repair_key = fx.make_view(
                    repair_key_ptr,
                    fx.make_layout((16, head_dim), (head_dim, 1)),
                )
                repair_score_ptr = fx.add_offset(fx.get_iter(s_score), repair_wave * 16)
                repair_score = fx.make_view(
                    repair_score_ptr,
                    fx.make_layout((16, 16), (TILE_TOKENS, 1)),
                )
                repair_thread = qk_mma.thr_slice(lane)
                repair_q = repair_thread.make_fragment_A(s_query)
                repair_k = repair_thread.make_fragment_B(repair_key)
                repair_acc = repair_thread.make_fragment_C(repair_score)
                repair_copy = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
                repair_copy_q = fx.make_tiled_copy_A(repair_copy, qk_mma).get_slice(
                    lane
                )
                fx.copy(
                    repair_copy,
                    repair_copy_q.partition_S(s_query),
                    repair_copy_q.retile(repair_q),
                )
                if const_expr(is_fp8):
                    repair_k.store(load_fp8_key_fragment(tile, repair_wave))
                else:
                    repair_copy_k = fx.make_tiled_copy_B(repair_copy, qk_mma).get_slice(
                        lane
                    )
                    fx.copy(
                        repair_copy,
                        repair_copy_k.partition_S(repair_key),
                        repair_copy_k.retile(repair_k),
                    )
                repair_acc.fill(0.0)
                for repair_k_repeat in range_constexpr(head_dim // 16):
                    fx.gemm(
                        qk_mma,
                        repair_acc,
                        repair_q[None, None, repair_k_repeat],
                        repair_k[None, None, repair_k_repeat],
                        repair_acc,
                    )
                if lane < 16:
                    repair_values = fx.Vector(repair_acc.load())
                    s_score[4, repair_wave * 16 + lane] = repair_values[4]
            lds_barrier()

            # A producer/repair wave pair owns one interleaved 16-token
            # partition.  The two waves advance two query rows at a time, so
            # all four partitions retain independent FP32 online-softmax state
            # while the complete CTA continues to synchronize the phase-reused
            # K/V arena.
            partition = wave % PARTITIONS
            pair_role = wave // PARTITIONS
            tile_has_tokens = fx.Int32(tile) + partition * 16 < end
            for row_round in range_constexpr(2):
                row = pair_role + row_round * 2
                live_row = row < query_group_size
                safe_row = live_row.select(row, fx.Int32(0))
                token_local = partition * 16 + lane
                token = fx.Int32(tile) + token_local
                valid = (lane < 16) & (token < end)
                score = valid.select(
                    fx.Float32(s_score[safe_row, token_local]), neg_inf
                )
                tile_max = _wave_reduce(score, "max")
                running_max = fx.Float32(s_max[partition, safe_row])
                candidate_max = fx.max(running_max, tile_max)
                new_max = tile_has_tokens.select(candidate_max, running_max)
                probability = valid.select(fmath.exp2((score - new_max) * LOG2E), zero)
                tile_sum = _wave_reduce(probability, "sum")
                alpha = tile_has_tokens.select(
                    fmath.exp2((running_max - new_max) * LOG2E),
                    fx.Float32(1.0),
                )
                if live_row:
                    if lane < 16:
                        s_prob[row, token_local] = probability.to(fx.BFloat16)
                    if lane == 0:
                        s_alpha[partition, row] = alpha
                        s_max[partition, row] = new_max
                        s_sum[partition, row] = (
                            fx.Float32(s_sum[partition, row]) * alpha + tile_sum
                        )

            # Phase-reuse Q/K storage for V once score reads are complete.
            lds_barrier()
            if const_expr(not is_fp8):
                for load_iter in range_constexpr(
                    (head_dim * TILE_TOKENS) // BLOCK_THREADS
                ):
                    linear = tid + load_iter * BLOCK_THREADS
                    d = linear // TILE_TOKENS
                    token_local = linear % TILE_TOKENS
                    token = fx.Int32(tile) + token_local
                    valid = token < end
                    safe_token = valid.select(token, fx.Int32(0))
                    logical_page = safe_token // page_size
                    in_page = safe_token - logical_page * page_size
                    physical_page = fx.Int32(
                        block_tables[seq * table_stride + logical_page]
                    )
                    value_index = (
                        physical_page * v_stride0
                        + kv_head * v_stride1
                        + d * v_stride2
                        + in_page * v_stride3
                    )
                    s_value[d, token_local] = valid.select(
                        value[value_index], fx.BFloat16(0.0)
                    )
                lds_barrier()

            # Each pair covers the complete D dimension in 64-column rounds.
            # Its WMMA result is merged immediately into the partition's FP32
            # LDS state, avoiding a large per-lane accumulator and all global
            # mid_out/mid_lse traffic.
            prob_ptr = fx.add_offset(fx.get_iter(s_prob), partition * 16)
            prob = fx.make_view(
                prob_ptr,
                fx.make_layout((16, 16), (TILE_TOKENS, 1)),
            )
            pv_thread = pv_mma.thr_slice(lane)
            copy_bf16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
            copy_p = fx.make_tiled_copy_A(copy_bf16, pv_mma).get_slice(lane)
            frag_p = pv_thread.make_fragment_A(prob)
            fx.copy(copy_bf16, copy_p.partition_S(prob), copy_p.retile(frag_p))
            for d_repeat in range_constexpr(head_dim // 64):
                d_base = pair_role * 32 + d_repeat * 64
                value_tile_ptr = fx.add_offset(
                    fx.get_iter(s_value),
                    d_base * TILE_TOKENS + partition * 16,
                )
                value_tile = fx.make_view(
                    value_tile_ptr,
                    fx.make_layout((32, 16), (TILE_TOKENS, 1)),
                )
                out_wave = fx.make_view(
                    fx.get_iter(s_score),
                    fx.make_layout((16, 32), (32, 1)),
                )
                frag_v = pv_thread.make_fragment_B(value_tile)
                frag_o = pv_thread.make_fragment_C(out_wave)
                if const_expr(is_fp8):
                    lane_column = lane % 16
                    token_half = (lane // 16) * 8
                    token_local = partition * 16 + token_half
                    token = fx.Int32(tile) + token_local
                    valid_start = token < end
                    safe_token = valid_start.select(token, fx.Int32(0))
                    logical_page = safe_token // page_size
                    in_page = safe_token - logical_page * page_size
                    physical_page = fx.Int32(
                        block_tables[seq * table_stride + logical_page]
                    )
                    for column_fragment in range_constexpr(2):
                        d = d_base + column_fragment * 16 + lane_column
                        value_index = (
                            physical_page * v_stride0
                            + kv_head * v_stride1
                            + d * v_stride2
                            + in_page * v_stride3
                        )
                        prefetch_value_u8x8(column_fragment, value_index)
                    fx.rocdl.s_wait_loadcnt(0)
                    fragment_values = []
                    for column_fragment in range_constexpr(2):
                        value_values = dequant_fp8x8(
                            consume_value_u8x8(column_fragment),
                            v_scale,
                        )
                        for token_inner in range_constexpr(8):
                            element_valid = token + token_inner < end
                            fragment_values.append(
                                element_valid.select(
                                    value_values[token_inner],
                                    fx.BFloat16(0.0),
                                )
                            )
                    frag_v.store(
                        fx.Vector.from_elements(
                            fragment_values,
                            dtype=fx.BFloat16,
                        )
                    )
                else:
                    copy_v = fx.make_tiled_copy_B(copy_bf16, pv_mma).get_slice(lane)
                    fx.copy(
                        copy_bf16,
                        copy_v.partition_S(value_tile),
                        copy_v.retile(frag_v),
                    )
                frag_o.fill(0.0)
                fx.gemm(
                    pv_mma,
                    frag_o,
                    frag_p,
                    frag_v,
                    frag_o,
                )
                out_values = fx.Vector(frag_o.load())
                if lane < 16:
                    for out_repeat in range_constexpr(2):
                        d = d_base + out_repeat * 16 + lane
                        for output_row in range_constexpr(query_group_size):
                            accum_index = out_repeat * 8 + output_row
                            s_partial[partition, output_row, d] = fx.Float32(
                                s_partial[partition, output_row, d]
                            ) * fx.Float32(s_alpha[partition, output_row]) + fx.Float32(
                                out_values[accum_index]
                            )
            lds_barrier()

        # One wave per live query row combines the four partition states using
        # the stable log-sum-exp rule, then writes the caller-owned output once.
        if wave < query_group_size:
            output_row = wave
            qh = kv_head * query_group_size + output_row
            global_max = fx.Float32(s_max[0, output_row])
            for partition_index in range_constexpr(1, PARTITIONS):
                global_max = fx.max(
                    global_max,
                    fx.Float32(s_max[partition_index, output_row]),
                )
            for d_iter in range_constexpr(head_dim // WAVE_SIZE):
                d = lane + d_iter * WAVE_SIZE
                combined_sum = zero
                combined_output = zero
                for partition_index in range_constexpr(PARTITIONS):
                    partition_sum = fx.Float32(s_sum[partition_index, output_row])
                    live_partition = partition_sum > zero
                    factor = live_partition.select(
                        fmath.exp2(
                            (
                                fx.Float32(s_max[partition_index, output_row])
                                - global_max
                            )
                            * LOG2E
                        ),
                        zero,
                    )
                    combined_sum = combined_sum + partition_sum * factor
                    combined_output = (
                        combined_output
                        + fx.Float32(s_partial[partition_index, output_row, d]) * factor
                    )
                normalized = combined_output / (combined_sum + 1.0e-10)
                out_index = seq * mo_stride0 + qh * mo_stride1 + d
                final_value = (end > begin).select(normalized, zero)
                if const_expr(is_fp16_query):
                    mid_out[out_index] = final_value.to(fx.Float16)
                else:
                    mid_out[out_index] = final_value.to(fx.BFloat16)

    @flyc.jit
    def launch(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        block_tables: fx.Tensor,
        seq_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        k_scale: fx.Tensor,
        v_scale: fx.Tensor,
        mid_out: fx.Tensor,
        mid_lse: fx.Tensor,
        batch: fx.Int32,
        table_stride: fx.Int32,
        q_stride0: fx.Int32,
        q_stride1: fx.Int32,
        k_stride0: fx.Int32,
        k_stride1: fx.Int32,
        k_stride2: fx.Int32,
        k_stride3: fx.Int32,
        k_stride4: fx.Int32,
        v_stride0: fx.Int32,
        v_stride1: fx.Int32,
        v_stride2: fx.Int32,
        v_stride3: fx.Int32,
        mo_stride0: fx.Int32,
        mo_stride1: fx.Int32,
        mo_stride2: fx.Int32,
        ml_stride0: fx.Int32,
        ml_stride1: fx.Int32,
        ml_stride2: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.BFloat16, fx.Float32))
        qk_mma = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        pv_mma = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        grouped_stage1(
            query,
            key,
            value,
            block_tables,
            seq_lens,
            query_start_loc,
            k_scale,
            v_scale,
            mid_out,
            mid_lse,
            batch,
            table_stride,
            q_stride0,
            q_stride1,
            k_stride0,
            k_stride1,
            k_stride2,
            k_stride3,
            k_stride4,
            v_stride0,
            v_stride1,
            v_stride2,
            v_stride3,
            mo_stride0,
            mo_stride1,
            mo_stride2,
            ml_stride0,
            ml_stride1,
            ml_stride2,
            qk_mma,
            pv_mma,
        ).launch(
            grid=(batch * num_kv_heads * splits, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch
