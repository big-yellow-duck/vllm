# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64, Stream

from .common import load_pack_128b as _load_pack
from .common import store_pack_128b as _store_pack


def _load_i64_acquire(addr_i64):
    return fx.rocdl.global_load(
        addr_i64,
        fx.Int64,
        alignment=8,
        memory_order=fx.rocdl.MemoryOrder.Acquire,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i64_release(addr_i64, value):
    fx.rocdl.global_store(
        addr_i64,
        value,
        alignment=8,
        memory_order=fx.rocdl.MemoryOrder.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _sleep_one():
    fx.rocdl.sleep(1)


def _add_bf16_pack(lhs_raw, rhs_raw):
    lhs = lhs_raw.bitcast(fx.BFloat16).to(fx.Float32)
    rhs = rhs_raw.bitcast(fx.BFloat16).to(fx.Float32)
    return (lhs + rhs).to(fx.BFloat16).bitcast(fx.Int32)


@cache
def make_full_launcher(*, blocks: int, threads: int):
    if blocks not in (1, 2, 4):
        raise ValueError(f"full kernel blocks must be 1, 2, or 4, got {blocks}")

    SharedStorage = fx.struct(
        type(
            "FullSharedStorage",
            (),
            {"__annotations__": {"ticket": fx.Array[fx.Int64, 1, 8]}},
        )
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def tp2_allreduce_bf16_full(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_ready_addr: Int64,
        peer_ready_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_ptr = lds.ticket.ptr
        ticket = fx.Int64(0)

        if tid == fx.Int32(0):
            if const_expr(blocks == 1):
                ticket = fx.Int64(_load_i64_acquire(local_ready_addr)) + fx.Int64(1)
            else:
                block_ready_addr = local_ready_addr + fx.Int64(bid) * fx.Int64(8)
                previous = fx.Int64(_load_i64_acquire(block_ready_addr))
                if bid == fx.Int32(0):
                    ticket = previous + fx.Int64(1)
                    _store_i64_release(local_launch_addr, ticket)
                else:
                    ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
                    while ticket <= previous:
                        _sleep_one()
                        ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
            fx.ptr_store(fx.Vector.from_elements([ticket], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket = fx.Vector(
            fx.ptr_load(
                ticket_ptr,
                result_type=fx.Vector.make_type(1, fx.Int64),
            )
        )[0]

        parity_bytes = (ticket & fx.Int64(1)) * slot_bytes
        local_slot_addr = local_slot_addr + parity_bytes
        peer_slot_addr = peer_slot_addr + parity_bytes
        pack_count = numel // fx.Int32(8)
        global_thread = bid * fx.Int32(threads) + tid
        global_stride = fx.Int32(blocks * threads)

        for pack in range(global_thread, pack_count, global_stride):
            _store_pack(local_slot_addr, pack, _load_pack(input_addr, pack))
        gpu.barrier()

        if tid == fx.Int32(0):
            if const_expr(blocks == 1):
                _store_i64_release(local_ready_addr, ticket)
                peer_ticket = fx.Int64(_load_i64_acquire(peer_ready_addr))
                while peer_ticket < ticket:
                    _sleep_one()
                    peer_ticket = fx.Int64(_load_i64_acquire(peer_ready_addr))
            else:
                local_block_addr = local_ready_addr + fx.Int64(bid) * fx.Int64(8)
                peer_block_addr = peer_ready_addr + fx.Int64(bid) * fx.Int64(8)
                _store_i64_release(local_block_addr, ticket)
                peer_ticket = fx.Int64(_load_i64_acquire(peer_block_addr))
                while peer_ticket < ticket:
                    _sleep_one()
                    peer_ticket = fx.Int64(_load_i64_acquire(peer_block_addr))
        gpu.barrier()

        for pack in range(global_thread, pack_count, global_stride):
            _store_pack(
                output_addr,
                pack,
                _add_bf16_pack(
                    _load_pack(input_addr, pack),
                    _load_pack(peer_slot_addr, pack),
                ),
            )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_full(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_ready_addr: Int64,
        peer_ready_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        tp2_allreduce_bf16_full(
            input_addr,
            output_addr,
            local_slot_addr,
            peer_slot_addr,
            slot_bytes,
            local_ready_addr,
            peer_ready_addr,
            local_launch_addr,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_full.func.__name__ = f"launch_tp2_bf16_full_b{blocks}_t{threads}"
    return launch_full


@cache
def make_pipeline_launcher(*, blocks: int, threads: int, chunk_packs: int):
    if blocks < 6 or blocks > 16:
        raise ValueError(f"pipeline blocks must be in [6, 16], got {blocks}")
    if chunk_packs < threads or chunk_packs % threads:
        raise ValueError("chunk_packs must be a positive multiple of threads")

    SharedStorage = fx.struct(
        type(
            "PipelineSharedStorage",
            (),
            {"__annotations__": {"ticket": fx.Array[fx.Int64, 1, 8]}},
        )
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def tp2_allreduce_bf16_pipeline(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_progress_addr: Int64,
        peer_progress_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_ptr = lds.ticket.ptr
        ticket = fx.Int64(0)

        if tid == fx.Int32(0):
            block_progress_addr = local_progress_addr + fx.Int64(bid) * fx.Int64(8)
            previous_progress = fx.Int64(_load_i64_acquire(block_progress_addr))
            previous_ticket = previous_progress >> fx.Int64(32)
            if bid == fx.Int32(0):
                ticket = fx.Int64(_load_i64_acquire(local_launch_addr)) + fx.Int64(1)
                _store_i64_release(local_launch_addr, ticket)
            else:
                ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
                while ticket <= previous_ticket:
                    _sleep_one()
                    ticket = fx.Int64(_load_i64_acquire(local_launch_addr))
            fx.ptr_store(fx.Vector.from_elements([ticket], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket = fx.Vector(
            fx.ptr_load(
                ticket_ptr,
                result_type=fx.Vector.make_type(1, fx.Int64),
            )
        )[0]

        parity_bytes = (ticket & fx.Int64(1)) * slot_bytes
        local_slot_addr = local_slot_addr + parity_bytes
        peer_slot_addr = peer_slot_addr + parity_bytes
        pack_count = numel // fx.Int32(8)
        first_pack = bid * fx.Int32(chunk_packs)
        round_stride = fx.Int32(blocks * chunk_packs)
        rounds = (pack_count - first_pack + round_stride - fx.Int32(1)) // round_stride

        for round_index in range(fx.Int32(0), rounds, fx.Int32(1)):
            chunk_begin = first_pack + round_index * round_stride
            candidate_end = chunk_begin + fx.Int32(chunk_packs)
            chunk_end = (candidate_end < pack_count).select(candidate_end, pack_count)

            for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                _store_pack(local_slot_addr, pack, _load_pack(input_addr, pack))
            gpu.barrier()

            if tid == fx.Int32(0):
                progress = (ticket << fx.Int64(32)) | fx.Int64(
                    round_index + fx.Int32(1)
                )
                local_block_addr = local_progress_addr + fx.Int64(bid) * fx.Int64(8)
                peer_block_addr = peer_progress_addr + fx.Int64(bid) * fx.Int64(8)
                _store_i64_release(local_block_addr, progress)
                peer_value = fx.Int64(_load_i64_acquire(peer_block_addr))
                while peer_value < progress:
                    _sleep_one()
                    peer_value = fx.Int64(_load_i64_acquire(peer_block_addr))
            gpu.barrier()

            for pack in range(chunk_begin + tid, chunk_end, fx.Int32(threads)):
                _store_pack(
                    output_addr,
                    pack,
                    _add_bf16_pack(
                        _load_pack(input_addr, pack),
                        _load_pack(peer_slot_addr, pack),
                    ),
                )

    flat_wg_size_attr = f"{threads},{threads}"

    @flyc.jit
    def launch_pipeline(
        input_addr: Int64,
        output_addr: Int64,
        local_slot_addr: Int64,
        peer_slot_addr: Int64,
        slot_bytes: Int64,
        local_progress_addr: Int64,
        peer_progress_addr: Int64,
        local_launch_addr: Int64,
        numel: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        tp2_allreduce_bf16_pipeline(
            input_addr,
            output_addr,
            local_slot_addr,
            peer_slot_addr,
            slot_bytes,
            local_progress_addr,
            peer_progress_addr,
            local_launch_addr,
            numel,
            value_attrs={"rocdl.flat_work_group_size": flat_wg_size_attr},
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_pipeline.func.__name__ = (
        f"launch_tp2_bf16_pipeline_b{blocks}_t{threads}_p{chunk_packs}"
    )
    return launch_pipeline


__all__ = ["make_full_launcher", "make_pipeline_launcher"]
