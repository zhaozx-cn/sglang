"""Triton-Ascend implementation of the LPLB ``lp_post`` operator.

GPU reference: ``triton_lplb/lp_post.py::lp_post_kernel`` — a single block
(``grid=(1,)``) runs four stages serially, relying on the implicit
``__syncthreads()`` between Triton loops:

    Stage 1  phy_prob[:] = 0
    Stage 2  phy_prob[phy_replicated] = max(x, 0)
    Stage 3  phy_prob[phy_single] = t1
    Stage 4  out[i] = phy_prob[wrap(log2phy[i])]

Architecture (see ``sketch.txt``):

* **No global scatter.**  Measured on Ascend910_9362, ``tl.store(ptr + idx_vec)``
  lowers to a per-lane scalar path (~0.13 us/lane) and gains almost nothing from
  more vector cores.  Stages 1-3 are therefore replaced by a *compare-sum
  inverse*: every physical slot sums the values of the source lanes that point
  at it.  Because ``phy_single`` and ``phy_replicated`` are disjoint, the result
  is bit-identical to sequential scatter, and there is no write conflict at all.
* **Two kernels.**  The table must be visible to every output program and there
  is no grid-wide barrier, so it has to go through global memory.
* **Stage 4 uses UB preload + ``tl.gather``** (index_computation card
  GATHER-L1.2 / LPLB-L1.2): the whole table (``PADP <= 1024`` elements = 4 KiB,
  far below the 32768-element UB limit) is loaded into UB once per program and
  selected with the hardware gather path.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FALLBACK_VEC_CORES = 40


def _num_vectorcore(device_index):
    """Number of AIV cores for the current device (host side only).

    Pure metadata lookup — no device-to-host tensor read.
    """
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
def build_tab_kernel(
    x_ptr, t1_ptr, ps_ptr, pr_ptr, tab_ptr,
    NR: tl.constexpr, NS: tl.constexpr, P: tl.constexpr,
    NCH: tl.constexpr, NC: tl.constexpr,
    PCH: tl.constexpr, SCH: tl.constexpr,
):
    """Equivalent of stages 1-3: build ``phy_prob`` (zero-init + two scatters).

    Each program owns ``PCH`` consecutive physical slots (contiguous write side,
    constraint G1) and accumulates the contributions of every source lane whose
    index matches the slot.  Slots that no source points at keep the
    zero-initialised value, which also covers the ``-1`` sink slot ``P - 1``.
    """
    pid = tl.program_id(0).to(tl.int32)
    o_c = tl.arange(0, PCH)
    o_s = tl.arange(0, SCH)

    n_single_chunks: tl.constexpr = (NS + SCH - 1) // SCH
    n_red_chunks: tl.constexpr = (NR + SCH - 1) // SCH

    for c in tl.static_range(0, (NCH + NC - 1) // NC):
        ci = pid + c * NC
        slots = ci * PCH + o_c
        acc = tl.zeros([PCH], tl.float32)

        # --- source 1: phy_single / t1 (stage 3, no clamp) ---
        for j in tl.static_range(0, n_single_chunks):
            src = j * SCH + o_s
            ms = src < NS
            tv = tl.load(t1_ptr + src, mask=ms, other=0.0)
            si = tl.load(ps_ptr + src, mask=ms, other=-1).to(tl.int32)  # L1.1: int32
            acc += tl.sum(
                tl.where(slots[:, None] == si[None, :], tv[None, :], 0.0), axis=1
            )

        # --- source 2: phy_replicated / max(x, 0) (stage 2) ---
        for j in tl.static_range(0, n_red_chunks):
            src = j * SCH + o_s
            mr = src < NR
            xv = tl.load(x_ptr + src, mask=mr, other=0.0)
            xv = tl.where(xv > 0.0, xv, 0.0)  # clamp(min=0)
            ri = tl.load(pr_ptr + src, mask=mr, other=-1).to(tl.int32)
            acc += tl.sum(
                tl.where(slots[:, None] == ri[None, :], xv[None, :], 0.0), axis=1
            )

        # Full coverage of [0, P): this replaces the explicit zero-init kernel.
        tl.store(tab_ptr + slots, acc, mask=(ci < NCH) & (slots < P))


@triton.jit
def gather_out_kernel(
    out_ptr, l2p_ptr, tab_ptr,
    P: tl.constexpr, W: tl.constexpr, TO: tl.constexpr,
    NB: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr,
):
    """Stage 4: ``out[i] = phy_prob[wrap(log2phy[i])]`` via UB gather."""
    pid = tl.program_id(0).to(tl.int32)

    # Whole table into UB once per program (<= 32768 elements, GATHER-L1.4).
    o_p = tl.arange(0, W)
    tab = tl.load(tab_ptr + o_p, mask=o_p < P, other=0.0)

    o = tl.arange(0, BLOCK)
    for i in tl.static_range(0, (NB + NC - 1) // NC):
        blk = pid + i * NC
        oo = blk * BLOCK + o
        m = (blk < NB) & (oo < TO)
        k = tl.load(l2p_ptr + oo, mask=m, other=0).to(tl.int32)  # G4 / L1.1
        kw = tl.where(k < 0, k + P, k)  # -1 -> sink slot P-1 (always zero)
        tl.store(out_ptr + oo, tl.gather(tab, kw, 0), mask=m)


class ModelNew(nn.Module):
    """Triton-Ascend port of ``lp_post_kernel``."""

    def __init__(self):
        super().__init__()
        self._vec_cores = None

    def forward(
        self,
        log2phy_prob: torch.Tensor,
        x: torch.Tensor,
        t1: torch.Tensor,
        phy_single: torch.Tensor,
        phy_replicated: torch.Tensor,
        log2phy: torch.Tensor,
    ):
        num_single = t1.shape[0]
        num_red_phy = x.shape[0]
        phy_prob_size = num_single + num_red_phy + 1
        total_out = log2phy.numel()

        if self._vec_cores is None:
            self._vec_cores = _num_vectorcore(x.device.index or 0)
        nvec = self._vec_cores

        pad_p = _next_pow2(phy_prob_size)
        tab = torch.empty(pad_p, dtype=torch.float32, device=x.device)

        # --- kernel 1: build the physical probability table ---
        pch = 16 if phy_prob_size <= 320 else 32
        nch = (phy_prob_size + pch - 1) // pch
        nc1 = min(nvec, nch)
        build_tab_kernel[(nc1,)](
            x, t1, phy_single, phy_replicated, tab,
            NR=num_red_phy, NS=num_single, P=phy_prob_size,
            NCH=nch, NC=nc1, PCH=pch, SCH=128,
        )

        # --- kernel 2: gather the logical outputs ---
        block = 512 if total_out <= 2560 else 1024
        nb = (total_out + block - 1) // block
        nc2 = min(nvec, nb)
        gather_out_kernel[(nc2,)](
            log2phy_prob, log2phy, tab,
            P=phy_prob_size, W=pad_p, TO=total_out,
            NB=nb, NC=nc2, BLOCK=block,
        )

        return log2phy_prob
