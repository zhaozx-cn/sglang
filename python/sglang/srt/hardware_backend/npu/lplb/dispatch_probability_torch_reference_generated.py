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
* **Small tables go to UB once per program + ``tl.gather``** (index_computation
  §4.6 L1.2 / §4.7 L1.6).  ``num_logical * max_copies <= 8192`` elements, so the
  whole probability table and the logical->physical table fit in UB; the
  per-channel selection then uses the hardware gather path instead of
  ``tl.load(prob_ptr + logical_id * MC + c)`` double-indirect addressing.
* **Sequential cumulative sum.**  The output is an integral expert id compared
  with strict equality, so the prefix sum is accumulated in ``c = 0..MC-1``
  order with ``tl.static_range`` -- exactly the reference's ``cumsum`` / CUDA
  kernel ``for`` loop order.  ``tl.cumsum`` (tree scan) is deliberately avoided.
* The map table is carried as fp32 inside the kernel (physical ids are far below
  2**24, so the cast is exact) because the Ascend gather path is the validated
  fp32 form; the value is cast back to the output dtype just before the store.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FALLBACK_VEC_CORES = 40


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
):
    """``out[i] = log2phy_map[topk_ids[i], inverse_cdf(log2phy_prob[topk_ids[i]])]``."""
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: whole small tables into UB, once per program ---------------
    ot = tl.arange(0, PADT)
    tm = ot < TAB
    pf = tl.load(prob_ptr + ot, mask=tm, other=0.0)
    mf = tl.load(map_ptr + ot, mask=tm, other=-1).to(tl.float32)
    vf = tl.where(mf >= 0.0, 1.0, 0.0)  # valid-replica indicator (fallback weights)

    o = tl.arange(0, BLOCK)
    for i in tl.static_range(0, (NB + NC - 1) // NC):
        blk = pid + i * NC
        offs = blk * BLOCK + o
        m = offs < N

        # int64 ids are truncated right after the load (index_computation §4.5 L1.1)
        lid = tl.load(topk_ptr + offs, mask=m, other=0).to(tl.int32)
        # optimization point 11: issue the random-value load up-front (no data
        # dependency on the gather chain) so MTE2 overlaps the vector work
        r0 = tl.load(rnd_ptr + offs, mask=m, other=0.0)
        base = lid * MC

        # --- stage 1: row sum over the probability row -----------------------
        row_sum = tl.zeros([BLOCK], tl.float32)
        nvalid = tl.zeros([BLOCK], tl.float32)
        for c in tl.static_range(0, MC):
            idx = base + c
            row_sum += tl.gather(pf, idx, 0)
            nvalid += tl.gather(vf, idx, 0)

        u = r0 * row_sum
        uf = r0 * nvalid
        has_p = row_sum > 0.0

        # --- stage 2: sequential inverse CDF (same order as the reference) ---
        cum = tl.zeros([BLOCK], tl.float32)
        cumf = tl.zeros([BLOCK], tl.float32)
        chosen = tl.zeros([BLOCK], tl.int32)
        for c in tl.static_range(0, MC):
            idx = base + c
            cum += tl.gather(pf, idx, 0)
            cumf += tl.gather(vf, idx, 0)
            # count of cum[c] <= u over the row; zero rows use the valid-replica
            # prefix count, matching the reference's fallback_probs + row_sum.
            hit = tl.where(
                has_p, (cum <= u).to(tl.int32), (cumf <= uf).to(tl.int32)
            )
            chosen += hit
        chosen = tl.where(chosen > MC - 1, MC - 1, chosen)  # clamp(max=MC-1)

        # --- stage 3: one gather picks the physical id ----------------------
        res = tl.gather(mf, base + chosen, 0)
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=m)


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

        block = min(1024, _next_pow2(max(16, n)))
        nb = (n + block - 1) // block
        nc = min(nvec, nb)
        tab = num_logical * max_copies

        dispatch_probability_kernel[(nc,)](
            flat_ids,
            prob,
            mp,
            rv,
            out,
            N=n,
            MC=max_copies,
            TAB=tab,
            PADT=_next_pow2(tab),
            BLOCK=block,
            NB=nb,
            NC=nc,
        )
        return out.view(original_shape)
