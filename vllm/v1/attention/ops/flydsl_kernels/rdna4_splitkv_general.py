# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""General FP8/FP16 RDNA4 SplitKV stage."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from .rdna4_splitkv_common import (
    BLOCK_THREADS,
    LOG2E,
    NUM_WAVES,
    TILE_TOKENS,
    WAVE_SIZE,
    _dequant_fp8x8,
    _flat_view,
    _wave_reduce,
)


@functools.lru_cache(maxsize=128)
def compile_native_tail_stage(
    *,
    query_dtype: str,
    kv_dtype: str,
    head_dim: int,
    splits: int,
    num_kv_heads: int,
    query_group_size: int,
    page_size: int,
    softmax_scale: float,
):
    """Build one eight-wave stage with two compile-time query slots per wave."""

    if query_dtype not in ("bf16", "fp16"):
        raise ValueError(f"query_dtype must be 'bf16' or 'fp16', got {query_dtype!r}")
    if kv_dtype not in ("fp16", "fp8", "fp8fnuz"):
        raise ValueError(
            f"kv_dtype must be 'fp16', 'fp8', or 'fp8fnuz', got {kv_dtype!r}"
        )
    if head_dim not in (128, 256):
        raise ValueError(f"head_dim must be 128 or 256, got {head_dim}")
    if num_kv_heads < 1:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if not 1 <= query_group_size <= 2 * NUM_WAVES:
        raise ValueError(
            f"query_group_size must be between 1 and {2 * NUM_WAVES}, "
            f"got {query_group_size}"
        )
    if page_size < 8 or page_size % 8:
        raise ValueError(f"page_size must be a positive multiple of 8, got {page_size}")
    is_fp8 = kv_dtype in ("fp8", "fp8fnuz")
    is_fp8fnuz = kv_dtype == "fp8fnuz"
    compute_type = fx.Float16 if query_dtype == "fp16" else fx.BFloat16
    output_repeats = head_dim // (NUM_WAVES * 16)
    direct_cache = is_fp8 or kv_dtype == "fp16"
    compact_lds = kv_dtype == "fp16"

    q_bytes = 16 * head_dim * 2
    k_bytes = TILE_TOKENS * head_dim * 2
    score_offset = k_bytes
    score_bytes = 16 * TILE_TOKENS * 4
    v_bytes = head_dim * TILE_TOKENS * 2
    prob_offset = v_bytes
    prob_bytes = 16 * TILE_TOKENS * 2
    alpha_offset = prob_offset + prob_bytes
    alpha_bytes = 16 * 4
    if compact_lds:
        score_offset = 0
        prob_offset = 0
        alpha_offset = prob_bytes
    q_offset = max(score_offset + score_bytes, alpha_offset + alpha_bytes) + 2048
    lds_bytes = q_offset + q_bytes

    @fx.struct
    class SharedStorage:
        words: fx.Array[fx.Int32, lds_bytes // 4, 16]

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
        mid_lse = _flat_view(mid_lse_ptr)

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

        def make_128b_prefetcher(tensor_ptr, elem_type, width, slots):
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_type)
            registers = [
                fx.make_rmem_tensor(fx.make_layout(width, 1), elem_type)
                for _ in range_constexpr(slots)
            ]
            flat = fx.make_view(fx.get_iter(tensor_ptr), fx.make_layout(1 << 30, 1))
            divided = fx.logical_divide(flat, fx.make_layout(1, 1))

            def prefetch(slot, element_offset):
                fx.copy(
                    atom, fx.slice(divided, (None, element_offset)), registers[slot]
                )

            def consume(slot):
                return fx.Vector(fx.memref_load_vec(registers[slot]))

            return prefetch, consume

        def make_128b_loader(tensor_ptr, elem_type, width):
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_type)
            register = fx.make_rmem_tensor(fx.make_layout(width, 1), elem_type)
            flat = fx.make_view(fx.get_iter(tensor_ptr), fx.make_layout(1 << 30, 1))
            divided = fx.logical_divide(flat, fx.make_layout(1, 1))

            def load(element_offset):
                fx.copy(atom, fx.slice(divided, (None, element_offset)), register)
                return fx.Vector(fx.memref_load_vec(register))

            return load

        if const_expr(is_fp8):
            load_key_u8x8 = make_64b_loader(key_ptr, fx.Uint8, 8, recast=True)
            value_window = 4 if is_fp8fnuz else 3
            prefetch_value_u8x8, consume_value_u8x8 = make_64b_prefetcher(
                value_ptr,
                fx.Uint8,
                8,
                value_window * output_repeats,
                recast=True,
            )
        if const_expr(direct_cache and not is_fp8):
            load_key_f16x8 = make_128b_loader(key_ptr, fx.Float16, 8)
            prefetch_value_f16x8, consume_value_f16x8 = make_128b_prefetcher(
                value_ptr, fx.Float16, 8, 4 * output_repeats
            )

        def dequant_fp8x8(raw, scale):
            return _dequant_fp8x8(raw, scale, compute_type, is_fp8fnuz=is_fp8fnuz)

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

        s_query = shared_view(q_offset, compute_type, (16, head_dim), (head_dim, 1))
        s_key = shared_view(0, compute_type, (TILE_TOKENS, head_dim), (head_dim, 1))
        s_score = shared_view(
            score_offset, fx.Float32, (16, TILE_TOKENS), (TILE_TOKENS, 1)
        )
        s_value = shared_view(
            0, compute_type, (head_dim, TILE_TOKENS), (TILE_TOKENS, 1)
        )
        s_prob = shared_view(
            prob_offset, compute_type, (16, TILE_TOKENS), (TILE_TOKENS, 1)
        )
        s_alpha = shared_view(alpha_offset, fx.Float32, (16,), (1,))

        def store_alpha(index, value):
            s_alpha[index] = value

        def load_key_fragment(tile_base, token_wave):
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
                if const_expr(is_fp8):
                    key_values = dequant_fp8x8(load_key_u8x8(key_index), k_scale)
                else:
                    key_values = load_key_f16x8(key_index)
                for value_index in range_constexpr(8):
                    fragment_values.append(
                        valid.select(key_values[value_index], compute_type(0.0))
                    )
            fx.rocdl.s_wait_loadcnt(0)
            return fx.Vector.from_elements(fragment_values, dtype=compute_type)

        # Q is invariant across context tiles. Pad ten inactive query rows with
        # zero so the 16-row WMMA tile remains regular.
        for q_iter in range_constexpr((16 * head_dim) // BLOCK_THREADS):
            linear = tid + q_iter * BLOCK_THREADS
            q_local = linear // head_dim
            d = linear % head_dim
            valid_q = q_local < query_group_size
            safe_q = valid_q.select(q_local, fx.Int32(0))
            qh = kv_head * query_group_size + safe_q
            q_value = query[query_row * q_stride0 + qh * q_stride1 + d]
            s_query[q_local, d] = valid_q.select(
                q_value.to(compute_type), compute_type(0.0)
            )
        lds_barrier()

        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        # Each wave owns two softmax rows, while its sixteen output accumulators
        # cover the two 16-column WMMA fragments.  Keeping only these scalars
        # loop-carried avoids the padded per-row/per-lane state that spills in
        # the generic large-GQA Triton stage.
        init_state = [neg_inf, zero, neg_inf, zero] + [
            zero for _ in range_constexpr(8 * output_repeats)
        ]

        for tile, state in range(begin, end, TILE_TOKENS, init=init_state):
            running_max = [fx.Float32(state[0]), fx.Float32(state[2])]
            running_sum = [fx.Float32(state[1]), fx.Float32(state[3])]
            accum = [
                fx.Float32(state[4 + j]) for j in range_constexpr(8 * output_repeats)
            ]

            # BF16 uses a shared K tile. FP8 is loaded directly into the native
            # WMMA B fragments below, avoiding a widened BF16 LDS handoff.
            if const_expr(not direct_cache):
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
                        key[key_index], compute_type(0.0)
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
                copy_bf16 = fx.make_copy_atom(fx.UniversalCopy16b(), compute_type)
                copy_q = fx.make_tiled_copy_A(copy_bf16, qk_mma).get_slice(lane)
                fx.copy(
                    copy_bf16,
                    copy_q.partition_S(s_query),
                    copy_q.retile(frag_q),
                )
                if const_expr(direct_cache):
                    frag_k.store(load_key_fragment(tile, wave))
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
                repair_copy = fx.make_copy_atom(fx.UniversalCopy16b(), compute_type)
                repair_copy_q = fx.make_tiled_copy_A(repair_copy, qk_mma).get_slice(
                    lane
                )
                fx.copy(
                    repair_copy,
                    repair_copy_q.partition_S(s_query),
                    repair_copy_q.retile(repair_q),
                )
                if const_expr(direct_cache):
                    repair_k.store(load_key_fragment(tile, repair_wave))
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

            # Two compile-time row slots per wave own all rows 0..15 exactly
            # once.  Both rows share the already-produced score tile and the
            # K/V tile, but carry independent FP32 online-softmax state.
            token0 = lane
            token1 = lane + WAVE_SIZE
            valid0 = fx.Int32(tile) + token0 < end
            valid1 = fx.Int32(tile) + token1 < end

            # Clear all probability rows before the owner waves publish them.
            for p_iter in range_constexpr((16 * TILE_TOKENS) // BLOCK_THREADS):
                linear = tid + p_iter * BLOCK_THREADS
                s_prob[linear // TILE_TOKENS, linear % TILE_TOKENS] = compute_type(0.0)

            new_max = []
            next_sum = []
            for row_slot in range_constexpr(2):
                owned_row = wave + row_slot * NUM_WAVES
                live_row = owned_row < query_group_size
                safe_row = live_row.select(owned_row, fx.Int32(0))
                score0 = valid0.select(
                    fx.Float32(s_score[safe_row, token0]) * softmax_scale,
                    neg_inf,
                )
                score1 = valid1.select(
                    fx.Float32(s_score[safe_row, token1]) * softmax_scale,
                    neg_inf,
                )
                score0 = live_row.select(score0, neg_inf)
                score1 = live_row.select(score1, neg_inf)
                tile_max = _wave_reduce(fx.max(score0, score1), "max")
                slot_max = fx.max(running_max[row_slot], tile_max)
                p0 = live_row.select(
                    valid0.select(fmath.exp2((score0 - slot_max) * LOG2E), zero),
                    zero,
                )
                p1 = live_row.select(
                    valid1.select(fmath.exp2((score1 - slot_max) * LOG2E), zero),
                    zero,
                )
                tile_sum = _wave_reduce(p0 + p1, "sum")
                alpha = live_row.select(
                    fmath.exp2((running_max[row_slot] - slot_max) * LOG2E),
                    zero,
                )
                new_max.append(slot_max)
                next_sum.append(running_sum[row_slot] * alpha + tile_sum)
                if live_row:
                    s_prob[owned_row, token0] = p0.to(compute_type)
                    s_prob[owned_row, token1] = p1.to(compute_type)
                    if lane == 0:
                        store_alpha(owned_row, alpha)

            # Phase-reuse Q/K storage for V once score reads are complete.
            lds_barrier()
            if const_expr(not direct_cache):
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
                        value[value_index], compute_type(0.0)
                    )
                lds_barrier()

            # Each wave independently computes a 16x32 output slice. Keeping
            # the wave tiling explicit avoids coupling the WMMA atom's wave32
            # register ABI to a multi-wave C-copy decomposition.
            value_wave_ptr = fx.add_offset(
                fx.get_iter(s_value),
                wave * (16 * output_repeats) * TILE_TOKENS,
            )
            value_wave = fx.make_view(
                value_wave_ptr,
                fx.make_layout((16 * output_repeats, TILE_TOKENS), (TILE_TOKENS, 1)),
            )
            out_wave_ptr = fx.get_iter(s_score)
            out_wave = fx.make_view(
                out_wave_ptr,
                fx.make_layout((16, 16 * output_repeats), (16 * output_repeats, 1)),
            )
            pv_thread = pv_mma.thr_slice(lane)
            frag_p = pv_thread.make_fragment_A(s_prob)
            frag_v = pv_thread.make_fragment_B(value_wave)
            frag_o = pv_thread.make_fragment_C(out_wave)
            copy_bf16 = fx.make_copy_atom(fx.UniversalCopy16b(), compute_type)
            copy_p = fx.make_tiled_copy_A(copy_bf16, pv_mma).get_slice(lane)
            fx.copy(copy_bf16, copy_p.partition_S(s_prob), copy_p.retile(frag_p))
            frag_o.fill(0.0)
            if const_expr(is_fp8 and not is_fp8fnuz):
                lane_column = lane % 16
                token_half = (lane // 16) * 8
                value_window = 3
                for seed_fragment in range_constexpr(value_window):
                    seed_token_local = seed_fragment * 16 + token_half
                    seed_token = fx.Int32(tile) + seed_token_local
                    seed_valid_start = seed_token < end
                    seed_safe_token = seed_valid_start.select(seed_token, fx.Int32(0))
                    seed_logical_page = seed_safe_token // page_size
                    seed_in_page = seed_safe_token - seed_logical_page * page_size
                    seed_physical_page = fx.Int32(
                        block_tables[seq * table_stride + seed_logical_page]
                    )
                    seed_slot_base = seed_fragment * output_repeats
                    for column_fragment in range_constexpr(output_repeats):
                        d = (
                            wave * (16 * output_repeats)
                            + column_fragment * 16
                            + lane_column
                        )
                        seed_value_index = (
                            seed_physical_page * v_stride0
                            + kv_head * v_stride1
                            + d * v_stride2
                            + seed_in_page * v_stride3
                        )
                        prefetch_value_u8x8(
                            seed_slot_base + column_fragment,
                            seed_value_index,
                        )

                for token_fragment in range_constexpr(TILE_TOKENS // 16):
                    token_local = token_fragment * 16 + token_half
                    slot_base = (token_fragment % value_window) * output_repeats
                    trailing_fragments = min(
                        value_window - 1,
                        TILE_TOKENS // 16 - token_fragment - 1,
                    )
                    fx.rocdl.s_wait_loadcnt(trailing_fragments * output_repeats)

                    fragment_values = []
                    for column_fragment in range_constexpr(output_repeats):
                        value_values = dequant_fp8x8(
                            consume_value_u8x8(slot_base + column_fragment),
                            v_scale,
                        )
                        for token_inner in range_constexpr(8):
                            element_valid = (
                                fx.Int32(tile) + token_local + token_inner < end
                            )
                            fragment_values.append(
                                element_valid.select(
                                    value_values[token_inner],
                                    compute_type(0.0),
                                )
                            )
                    frag_v[None, None, token_fragment].store(
                        fx.Vector.from_elements(fragment_values, dtype=compute_type)
                    )
                    if const_expr(token_fragment + value_window < TILE_TOKENS // 16):
                        next_token_local = (
                            token_fragment + value_window
                        ) * 16 + token_half
                        next_token = fx.Int32(tile) + next_token_local
                        next_valid_start = next_token < end
                        next_safe_token = next_valid_start.select(
                            next_token, fx.Int32(0)
                        )
                        next_logical_page = next_safe_token // page_size
                        next_in_page = next_safe_token - next_logical_page * page_size
                        next_physical_page = fx.Int32(
                            block_tables[seq * table_stride + next_logical_page]
                        )
                        for column_fragment in range_constexpr(output_repeats):
                            d = (
                                wave * (16 * output_repeats)
                                + column_fragment * 16
                                + lane_column
                            )
                            next_value_index = (
                                next_physical_page * v_stride0
                                + kv_head * v_stride1
                                + d * v_stride2
                                + next_in_page * v_stride3
                            )
                            prefetch_value_u8x8(
                                slot_base + column_fragment,
                                next_value_index,
                            )
                    fx.gemm(
                        pv_mma,
                        frag_o,
                        frag_p[None, None, token_fragment],
                        frag_v[None, None, token_fragment],
                        frag_o,
                    )
            else:
                if const_expr(direct_cache):
                    lane_column = lane % 16
                    token_half = (lane // 16) * 8
                    fragment_values = []
                    # Preserve the incumbent whole-fragment schedule for FP16
                    # cache and FNUZ FP8.  Only native OCP FP8 is pipelined.
                    for token_fragment in range_constexpr(TILE_TOKENS // 16):
                        token_local = token_fragment * 16 + token_half
                        token = fx.Int32(tile) + token_local
                        valid_start = token < end
                        safe_token = valid_start.select(token, fx.Int32(0))
                        logical_page = safe_token // page_size
                        in_page = safe_token - logical_page * page_size
                        physical_page = fx.Int32(
                            block_tables[seq * table_stride + logical_page]
                        )
                        for column_fragment in range_constexpr(output_repeats):
                            d = (
                                wave * (16 * output_repeats)
                                + column_fragment * 16
                                + lane_column
                            )
                            value_index = (
                                physical_page * v_stride0
                                + kv_head * v_stride1
                                + d * v_stride2
                                + in_page * v_stride3
                            )
                            if const_expr(is_fp8):
                                prefetch_value_u8x8(
                                    token_fragment * output_repeats + column_fragment,
                                    value_index,
                                )
                            else:
                                prefetch_value_f16x8(
                                    token_fragment * output_repeats + column_fragment,
                                    value_index,
                                )
                    if const_expr(is_fp8):
                        fx.rocdl.s_wait_loadcnt(0)
                    else:
                        fx.rocdl.s_wait_loadcnt(3 * output_repeats)
                    for token_fragment in range_constexpr(TILE_TOKENS // 16):
                        token_local = token_fragment * 16 + token_half
                        for column_fragment in range_constexpr(output_repeats):
                            if const_expr(is_fp8):
                                value_values = dequant_fp8x8(
                                    consume_value_u8x8(
                                        token_fragment * output_repeats
                                        + column_fragment
                                    ),
                                    v_scale,
                                )
                            else:
                                value_values = consume_value_f16x8(
                                    token_fragment * output_repeats + column_fragment
                                )
                            for token_inner in range_constexpr(8):
                                element_valid = (
                                    fx.Int32(tile) + token_local + token_inner < end
                                )
                                fragment_values.append(
                                    element_valid.select(
                                        value_values[token_inner],
                                        compute_type(0.0),
                                    )
                                )
                    frag_v.store(
                        fx.Vector.from_elements(fragment_values, dtype=compute_type)
                    )
                else:
                    copy_v = fx.make_tiled_copy_B(copy_bf16, pv_mma).get_slice(lane)
                    fx.copy(
                        copy_bf16,
                        copy_v.partition_S(value_wave),
                        copy_v.retile(frag_v),
                    )
                for k_repeat in range_constexpr(TILE_TOKENS // 16):
                    fx.gemm(
                        pv_mma,
                        frag_o,
                        frag_p[None, None, k_repeat],
                        frag_v[None, None, k_repeat],
                        frag_o,
                    )
            out_values = fx.Vector(frag_o.load())
            out_row_base = (lane // 16) * 8
            next_accum = []
            for out_repeat in range_constexpr(output_repeats):
                for out_row in range_constexpr(8):
                    accum_index = out_repeat * 8 + out_row
                    output_row = out_row_base + out_row
                    row_live = output_row < query_group_size
                    safe_row = row_live.select(output_row, fx.Int32(0))
                    next_accum.append(
                        row_live.select(
                            accum[accum_index] * fx.Float32(s_alpha[safe_row])
                            + out_values[accum_index],
                            zero,
                        )
                    )
            lds_barrier()
            results = (
                yield [new_max[0], next_sum[0], new_max[1], next_sum[1]] + next_accum
            )

        has_tokens = end > begin
        for row_slot in range_constexpr(2):
            owned_row = wave + row_slot * NUM_WAVES
            if owned_row < query_group_size:
                qh = kv_head * query_group_size + owned_row
                final_sum = fx.Float32(results[row_slot * 2 + 1])
                if lane == 0:
                    store_alpha(owned_row, final_sum)
                    lse_index = seq * ml_stride0 + qh * ml_stride1 + split * ml_stride2
                    lse = (
                        fx.Float32(results[row_slot * 2])
                        + fmath.log2(final_sum) / LOG2E
                    )
                    mid_lse[lse_index] = has_tokens.select(lse, neg_inf)
        lds_barrier()
        # Each lane half owns eight query rows; both halves now publish their
        # exact rows instead of indexing a padded 16-row accumulator array.
        for out_repeat in range_constexpr(output_repeats):
            d = wave * (16 * output_repeats) + out_repeat * 16 + lane % 16
            for local_row in range_constexpr(8):
                output_row = (lane // 16) * 8 + local_row
                if output_row < query_group_size:
                    qh = kv_head * query_group_size + output_row
                    out_index = (
                        seq * mo_stride0 + qh * mo_stride1 + split * mo_stride2 + d
                    )
                    accum_index = 4 + out_repeat * 8 + local_row
                    normalized = fx.Float32(results[accum_index]) / (
                        fx.Float32(s_alpha[output_row]) + 1.0e-10
                    )
                    mid_out[out_index] = has_tokens.select(normalized, zero)

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
        atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, compute_type, fx.Float32))
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
