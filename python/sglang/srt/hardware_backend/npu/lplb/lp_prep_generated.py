"""Triton Ascend implementation of the LPLB ``lp_prep`` fused kernel.

Phase 4 / opt_iter_2: optimization point #31 (IR analysis, TTIR-driven).

TTIR evidence (``lp_prep_kernel.ttir`` dumped from opt_iter_1):
  ``scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 { ... }``
  For NUM_GPUS=32 / GCHUNK=16 the matvec body runs as a 2-trip dynamic loop, so
  every trip re-materialises ``%57 = g0*16 + arange(16)``, ``%58 = %57 < 32``,
  the two pointer splats/addptrs and a fresh 16x512 tile. The trip count is a
  compile-time constant (``NCHUNK``), so the loop is pure overhead.

Applied change vs opt_iter_1:
  * ``GCHUNK`` is now sized to the row count instead of a fixed 16
    (``next_pow2(max(16, NUM_GPUS))``, with a UB budget guard that falls back to
    16 when ``GCHUNK*BS`` would exceed 96 KiB of f32), so ``NCHUNK == 1`` for
    every realistic LPLB shape and the row mask becomes compile-time uniform.
  * the chunk loop uses ``tl.static_range`` so the remaining iterations are
    unrolled in the frontend -- no ``scf.for`` survives into TTIR.

Earlier rounds (kept):
  * #4  discrete-access fix: ``global_counts`` is loaded contiguously into UB
    once and the random lookups become in-UB ``tl.gather``.
  * #11 load reorder: ``A_base_row_sum`` slice is issued before the big B1 tile.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lp_prep_kernel(
    A_full_ptr,
    b_ptr,
    t1_ptr,
    global_counts_ptr,
    log_single_ptr,
    log_replicated_ptr,
    B1_ptr,
    A_base_row_sum_ptr,
    NUM_SINGLE: tl.constexpr,
    NUM_RED_LOG: tl.constexpr,
    NUM_GPUS: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    NUM_LOGICAL: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
    BT: tl.constexpr,
    GCHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    # ---------------- Stage 1: gather counts, total, inv_total ----------------
    o_s = tl.arange(0, BS)
    m_s = o_s < NUM_SINGLE
    o_r = tl.arange(0, BR)
    m_r = o_r < NUM_RED_LOG

    # int64 logical ids -> int32 before entering the vector path.
    idx_s = tl.load(log_single_ptr + o_s, mask=m_s, other=0).to(tl.int32)
    idx_r = tl.load(log_replicated_ptr + o_r, mask=m_r, other=0).to(tl.int32)

    # Discrete-access fix (#4): the count table is small -> load it contiguously
    # into UB once, then select with in-UB gather instead of random global reads.
    o_t = tl.arange(0, BT)
    m_t = o_t < NUM_LOGICAL
    counts_tab = tl.load(global_counts_ptr + o_t, mask=m_t, other=0.0)

    c_s = tl.gather(counts_tab, idx_s, 0)
    c_r = tl.gather(counts_tab, idx_r, 0)
    c_s = tl.where(m_s, c_s, 0.0)
    c_r = tl.where(m_r, c_r, 0.0)

    # Masked lanes contribute exactly 0 to the sums (padding never enters exp).
    sum_s = tl.sum(c_s, axis=0)
    sum_r = tl.sum(c_r, axis=0)
    total = sum_s + sum_r
    inv_total = 1.0 / tl.maximum(total, 1.0)

    # ---------------- Stage 2: scale t1 and b[:NUM_RED_LOG] ----------------
    t1_s = c_s * inv_total
    b1_s = c_r * inv_total

    tl.store(t1_ptr + o_s, t1_s, mask=m_s)
    tl.store(b_ptr + o_r, b1_s, mask=m_r)

    # ---------------- Stage 4a: A_full[:NUM_RED_LOG, NV-1] ----------------
    base_r = tl.load(A_base_row_sum_ptr + o_r, mask=m_r, other=0.0)
    tl.store(A_full_ptr + o_r * NV + (NV - 1), b1_s - base_r, mask=m_r)

    # ---------------- Stage 3 + 4b: b2 = -(B1 @ t1), GPU row chunks ----------------
    # #31: NCHUNK is a compile-time constant -> unroll, no scf.for in TTIR.
    o_g = tl.arange(0, GCHUNK)
    for g0 in tl.static_range(0, NCHUNK):
        rows = g0 * GCHUNK + o_g
        m_g = rows < NUM_GPUS

        # Optimization point #11 (load order): this load only depends on the loop
        # variable, so emit it before the big B1 tile load.
        base_g = tl.load(A_base_row_sum_ptr + NUM_RED_LOG + rows, mask=m_g, other=0.0)

        b1_tile = tl.load(
            B1_ptr + rows[:, None] * NUM_SINGLE + o_s[None, :],
            mask=m_g[:, None] & m_s[None, :],
            other=0.0,
        )
        # Padded columns of t1_s are 0, so they add nothing to the row dot.
        part = tl.sum(b1_tile * t1_s[None, :], axis=1)
        b2 = -part

        tl.store(b_ptr + NUM_RED_LOG + rows, b2, mask=m_g)

        tl.store(
            A_full_ptr + (NUM_RED_LOG + rows) * NV + (NV - 1),
            b2 - base_g,
            mask=m_g,
        )


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


class ModelNew(nn.Module):
    """Drop-in replacement of the GPU Triton ``lp_prep`` wrapper."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        A_full: torch.Tensor,
        b: torch.Tensor,
        t1: torch.Tensor,
        global_counts: torch.Tensor,
        log_single: torch.Tensor,
        log_replicated: torch.Tensor,
        B1: torch.Tensor,
        A_base_row_sum: torch.Tensor,
    ):
        NC, NV = A_full.shape
        NUM_SINGLE = t1.shape[0]
        NUM_RED_LOG = log_replicated.shape[0]
        NUM_GPUS = B1.shape[0]
        NUM_LOGICAL = global_counts.shape[0]

        assert NC == NUM_RED_LOG + NUM_GPUS, "NC must equal NUM_RED_LOG + NUM_GPUS"
        assert B1.shape == (NUM_GPUS, NUM_SINGLE)

        BS = _next_pow2(max(16, NUM_SINGLE))
        BR = _next_pow2(max(16, NUM_RED_LOG))
        BT = _next_pow2(max(16, NUM_LOGICAL))

        # #31: size the row chunk to the GPU count so the matvec is a single
        # unrolled tile; keep the tile inside the UB budget (f32, <= 96 KiB).
        GCHUNK = _next_pow2(max(16, NUM_GPUS))
        if GCHUNK * BS > (96 * 1024) // 4:
            GCHUNK = 16
        NCHUNK = triton.cdiv(NUM_GPUS, GCHUNK)

        lp_prep_kernel[(1,)](
            A_full,
            b,
            t1,
            global_counts,
            log_single,
            log_replicated,
            B1,
            A_base_row_sum,
            NUM_SINGLE=NUM_SINGLE,
            NUM_RED_LOG=NUM_RED_LOG,
            NUM_GPUS=NUM_GPUS,
            NC=NC,
            NV=NV,
            NUM_LOGICAL=NUM_LOGICAL,
            BS=BS,
            BR=BR,
            BT=BT,
            GCHUNK=GCHUNK,
            NCHUNK=NCHUNK,
        )
        return A_full, b, t1
