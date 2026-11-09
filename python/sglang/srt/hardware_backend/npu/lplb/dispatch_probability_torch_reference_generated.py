"""Triton-Ascend implementation of the LPLB ``dispatch_probability`` operator.

Pure-torch reference (``triton_lplb/lplb.py::dispatch_probability_torch_reference``):

    probs = log2phy_prob[topk_ids]                 # gather row per lane
    maps  = log2phy_map[topk_ids]
    probs = where(probs.sum(-1) > 0, probs, (maps >= 0))   # zero-row fallback
    u     = random_vals * probs.sum(-1)
    chosen = (probs.cumsum(-1) <= u).sum(-1).clamp(max=MC-1)
    out   = maps.gather(1, chosen)                 # -> physical expert id

Architecture (see ``sketch.txt``):

* **Single fused kernel, lane-parallel over N = topk_ids.numel().**  Every lane
  owns one ``(token, slot)`` and performs the whole inverse-CDF sample in
  registers, so the ~8 torch ops of the reference collapse into one launch
  (index_computation G1: parallel axis follows the contiguous output/index).
* **Three table access paths, selected on the host by UB budget.**
  - ``PROB_UB=1, MAP_UB=1`` (both tables fit): the probability table and the
    logical->physical table are preloaded into UB once per program and the
    per-lane selection uses ``tl.gather`` (index_computation §4.6 L1.2 /
    §4.7 L1.6).
  - ``PROB_UB=1, MAP_UB=0`` (only one table fits): keep the probability table
    in UB (it is read twice per element) and read the map from GM.  Measured
    2.4x faster than the all-GM path at a 32K-entry table.
  - ``PROB_UB=0, MAP_UB=0`` (table too large even for one copy): every element
    is read from GM.  The compiler CSEs the duplicated reads across the two
    passes, so this path issues ``2*MC + 1`` scalar loads.
  The validity indicator is derived from the map value in every path, so no
  third table is ever live (the earlier three-table form sat right at the UB
  limit and failed to compile at PADT >= 16384).
* **Adaptive tile width.**  ``tl.gather`` is priced per instruction, not per
  lane, so BLOCK only affects the UB temporaries of the unrolled gather chain.
  The host therefore walks ``_BLOCK_LADDER`` (1024 -> 64) and keeps the UB path
  alive by narrowing the tile when the tables are large
  (``_ub_need(pad_tab, MC, BLOCK, tables) <= _UB_BUDGET_BYTES``).  Narrowing to
  64 matters once a row has many copies: it is what still lets *both* tables
  stay resident at MC=68 with a 16K-entry table (measured 0.65 ms vs 1.79 ms
  when the map falls back to GM reads).  Only tables whose single copy cannot
  fit at BLOCK=64 (>= 128K entries) take the all-GM path; two copies stop
  fitting at >= 32K entries, which is where the prob-only path takes over.
* **Bounded row unrolling.**  A row with ``MC`` copies costs ``2*MC+1`` unlifted
  ``tl.gather`` ops when the ``c`` loop is unrolled, and bishengir's
  auto-blockify pass is super-linear in that count: MC=68 never finished
  compiling (>420 s at BLOCK=128) and MC=20 took 97 s.  Rows wider than
  ``_UNROLL_MAX`` are therefore walked by a runtime ``tl.range`` with
  ``loop_unroll_factor=_CHUNK_UNROLL``, which keeps the body at a constant op
  count (MC=68: 20 s / 0.65 ms).  Rows at or below ``_UNROLL_MAX`` keep the
  fully unrolled form, so their codegen is unchanged.
* **Sequential cumulative sum.**  The output is an integral expert id compared
  with strict equality, so the prefix sum is accumulated in ``c = 0..MC-1``
  order with ``tl.static_range`` -- exactly the reference's ``cumsum`` / CUDA
  kernel ``for`` loop order.  ``tl.cumsum`` (tree scan) is deliberately avoided.
* The map table is carried as fp32 inside the UB path (physical ids are far
  below 2**24, so the cast is exact) because the Ascend gather path is the
  validated fp32 form; the GM path keeps ids as int and stores them directly.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FALLBACK_VEC_CORES = 40

# Ascend 910_9362 UB is 192 KiB; keep 16 KiB of slack for allocator rounding and
# the auto multi-buffer pass.
_UB_BUDGET_BYTES = 176 * 1024

# Candidate tile widths, largest first.  A larger BLOCK costs UB temporaries
# (~4*BLOCK bytes per live vector in the unrolled gather chain) but nothing else,
# because `tl.gather` is priced per instruction, not per lane.
_BLOCK_LADDER = (1024, 512, 256, 128, 64)
# BLOCK used when no table fits in UB (direct GM path): the per-lane indirect
# load is scalarised, so a narrow tile amortises it best.
_GM_BLOCK = 256

# Row widths up to this are fully unrolled over the MC copies.  Beyond it the
# row is walked by a *runtime* loop with a bounded unroll factor: unrolling all
# MC copies makes bishengir's auto-blockify pass super-linear (measured on
# n=12000: MC=68 never finishes, >420 s at BLOCK=128, and MC=20 takes 97 s),
# while the bounded form keeps the same op count per loop body by a wide margin
# (MC=68: 15.6 s / 2.44 ms vs torch 41.7 ms).
_UNROLL_MAX: tl.constexpr = 8
_CHUNK_UNROLL: tl.constexpr = 4


def _ub_need(pad_tab, mc, block, tables):
    """Bytes of UB: `tables` fp32 tables plus the live vectors of the gather chain."""
    return 4 * tables * pad_tab + 4 * block * (2 * mc + 10)


def _pick_block(pad_tab, mc, n, tables):
    """Largest ladder width whose UB footprint fits, or None if nothing fits."""
    top = min(1024, _next_pow2(max(16, n)))
    for cand in (top,) + tuple(b for b in _BLOCK_LADDER if b < top):
        if _ub_need(pad_tab, mc, cand, tables) <= _UB_BUDGET_BYTES:
            return cand
    return None


def _num_vectorcore(device_index):
    """Number of AIV cores for the current device (host-side metadata only)."""
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(device_index)
        n = props["num_vectorcore"]
        if n > 0:
            return n
    except Exception:
        pass
    return _FALLBACK_VEC_CORES


def _next_pow2(n, floor=16):
    p = floor
    while p < n:
        p *= 2
    return p


@triton.jit
def _row_at(p_src, m_src, prob_ptr, map_ptr, idx, msk,
            PROB_UB: tl.constexpr, MAP_UB: tl.constexpr):
    """One ``(prob, valid)`` pair of a row, from UB tables or from GM."""
    if PROB_UB:
        pv = tl.gather(p_src, idx, 0)
    else:
        pv = tl.load(prob_ptr + idx, mask=msk, other=0.0)
    if MAP_UB:
        mv = tl.gather(m_src, idx, 0)
    else:
        mv = tl.load(map_ptr + idx, mask=msk, other=-1).to(tl.float32)
    return pv, mv


@triton.jit
def dispatch_probability_kernel(
    topk_ptr,
    prob_ptr,
    map_ptr,
    rnd_ptr,
    out_ptr,
    N: tl.constexpr,
    MC: tl.constexpr,
    TAB: tl.constexpr,
    PADT: tl.constexpr,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    NC: tl.constexpr,
    PROB_UB: tl.constexpr,
    MAP_UB: tl.constexpr,
):
    """``out[i] = log2phy_map[topk_ids[i], inverse_cdf(log2phy_prob[topk_ids[i]])]``."""
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: whole tables into UB, once per program ---------------------
    if PROB_UB or MAP_UB:
        ot = tl.arange(0, PADT)
        tm = ot < TAB
    if PROB_UB:
        p_tab = tl.load(prob_ptr + ot, mask=tm, other=0.0)
    else:
        p_tab = prob_ptr  # placeholder: only read when PROB_UB
    if MAP_UB:
        m_tab = tl.load(map_ptr + ot, mask=tm, other=-1).to(tl.float32)
    else:
        m_tab = map_ptr  # placeholder: only read when MAP_UB

    o = tl.arange(0, BLOCK)
    for i in tl.static_range(0, (NB + NC - 1) // NC):
        blk = pid + i * NC
        offs = blk * BLOCK + o
        msk = offs < N

        # int64 ids are truncated right after the load (index_computation §4.5 L1.1)
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # optimization point 11: issue the random-value load up-front (no data
        # dependency on the gather chain) so MTE2 overlaps the vector work
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        base = lid * MC

        # --- stage 1: row sum over the probability row -----------------------
        row_sum = tl.zeros([BLOCK], tl.float32)
        nvalid = tl.zeros([BLOCK], tl.float32)
        if MC <= _UNROLL_MAX:
            for c in tl.static_range(0, MC):
                idx = base + c
                if PROB_UB:
                    pv = tl.gather(p_tab, idx, 0)
                else:
                    pv = tl.load(prob_ptr + idx, mask=msk, other=0.0)
                if MAP_UB:
                    mv = tl.gather(m_tab, idx, 0)
                else:
                    mv = tl.load(map_ptr + idx, mask=msk, other=-1).to(tl.float32)
                row_sum += pv
                nvalid += tl.where(mv >= 0.0, 1.0, 0.0)
        else:
            for c in tl.range(0, MC, loop_unroll_factor=_CHUNK_UNROLL):
                pv, mv = _row_at(p_tab, m_tab, prob_ptr, map_ptr, base + c, msk,
                                 PROB_UB, MAP_UB)
                row_sum += pv
                nvalid += tl.where(mv >= 0.0, 1.0, 0.0)

        u = r0 * row_sum
        uf = r0 * nvalid
        has_p = row_sum > 0.0

        # --- stage 2: sequential inverse CDF (same order as the reference) ---
        cum = tl.zeros([BLOCK], tl.float32)
        cumf = tl.zeros([BLOCK], tl.float32)
        chosen = tl.zeros([BLOCK], tl.int32)
        if MC <= _UNROLL_MAX:
            for c in tl.static_range(0, MC):
                idx = base + c
                if PROB_UB:
                    pv = tl.gather(p_tab, idx, 0)
                else:
                    pv = tl.load(prob_ptr + idx, mask=msk, other=0.0)
                if MAP_UB:
                    mv = tl.gather(m_tab, idx, 0)
                else:
                    mv = tl.load(map_ptr + idx, mask=msk, other=-1).to(tl.float32)
                cum += pv
                cumf += tl.where(mv >= 0.0, 1.0, 0.0)
                # count of cum[c] <= u over the row; zero rows use the valid-replica
                # prefix count, matching the reference's fallback_probs + row_sum.
                hit = tl.where(
                    has_p, (cum <= u).to(tl.int32), (cumf <= uf).to(tl.int32)
                )
                chosen += hit
        else:
            for c in tl.range(0, MC, loop_unroll_factor=_CHUNK_UNROLL):
                pv, mv = _row_at(p_tab, m_tab, prob_ptr, map_ptr, base + c, msk,
                                 PROB_UB, MAP_UB)
                cum += pv
                cumf += tl.where(mv >= 0.0, 1.0, 0.0)
                hit = tl.where(
                    has_p, (cum <= u).to(tl.int32), (cumf <= uf).to(tl.int32)
                )
                chosen += hit
        chosen = tl.where(chosen > MC - 1, MC - 1, chosen)  # clamp(max=MC-1)

        # --- stage 3: one lookup picks the physical id -----------------------
        if MAP_UB:
            res = tl.gather(m_tab, base + chosen, 0).to(out_ptr.dtype.element_ty)
        else:
            res = tl.load(map_ptr + base + chosen, mask=msk, other=-1).to(
                out_ptr.dtype.element_ty
            )
        tl.store(out_ptr + offs, res, mask=msk)


class ModelNew(nn.Module):
    """Triton-Ascend port of ``dispatch_probability_torch_reference``."""

    def __init__(self):
        super().__init__()
        self._vec_cores = None

    def forward(
        self,
        topk_ids: torch.Tensor,
        log2phy_prob: torch.Tensor,
        log2phy_map: torch.Tensor,
        random_vals: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = topk_ids.shape
        flat_ids = topk_ids.reshape(-1)
        n = flat_ids.numel()
        num_logical, max_copies = log2phy_prob.shape

        if self._vec_cores is None:
            self._vec_cores = _num_vectorcore(topk_ids.device.index or 0)
        nvec = self._vec_cores

        prob = log2phy_prob.contiguous()
        mp = log2phy_map.contiguous()
        rv = random_vals.contiguous()
        out = torch.empty(n, dtype=topk_ids.dtype, device=topk_ids.device)

        # Preloading is worth 1-2 orders of magnitude over per-lane GM reads, so
        # keep as many tables resident as the UB budget allows.  Beyond it the
        # compiler would abort with "ub overflow" (see the runbook).
        tab = num_logical * max_copies
        pad_tab = _next_pow2(tab)
        prob_ub = map_ub = False
        picked = _pick_block(pad_tab, max_copies, n, 2)
        if picked is not None:
            block, prob_ub, map_ub = picked, True, True
        else:
            picked = _pick_block(pad_tab, max_copies, n, 1)
            if picked is not None:
                block, prob_ub = picked, True
            else:
                block = min(_GM_BLOCK, _next_pow2(max(16, n)))
        nb = (n + block - 1) // block
        nc = min(nvec, nb)

        dispatch_probability_kernel[(nc,)](
            flat_ids,
            prob,
            mp,
            rv,
            out,
            N=n,
            MC=max_copies,
            TAB=tab,
            PADT=pad_tab,
            BLOCK=block,
            NB=nb,
            NC=nc,
            PROB_UB=prob_ub,
            MAP_UB=map_ub,
        )
        return out.view(original_shape)
