"""Ascend NPU backend for the LPLB fused operators.

The Triton-Ascend kernels in this package are ports of the CUDA JIT operators
in ``sglang.kernels.jit.csrc.lplb``; each generated module exposes a
``ModelNew`` wrapper.  This module adapts those wrappers to the same function
surface as :mod:`sglang.kernels.ops.lplb.cuda_solver` so that ``LPLBSolver``
and the probability dispatcher can select a backend per platform without
knowing any launch details.

Only the functions used on the LPLB hot path are exposed.  They intentionally
keep the CUDA signatures so the call sites stay backend-agnostic.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_NUM_ITERS = 5

# ``device`` is part of the key only to make repeated calls for a different
# device cheap; in practice every rank has one NPU.
_WARMED_UP: set[tuple[int, int, int, str]] = set()


@lru_cache(maxsize=None)
def _solve_module(num_iters: int):
    from sglang.srt.hardware_backend.npu.lplb.solve_ipm_torch_reference_generated import (
        ModelNew,
    )

    return ModelNew(num_iters=num_iters)


@lru_cache(maxsize=None)
def _prep_module():
    from sglang.srt.hardware_backend.npu.lplb.lp_prep_generated import ModelNew

    return ModelNew()


@lru_cache(maxsize=None)
def _post_module():
    from sglang.srt.hardware_backend.npu.lplb.lp_post_generated import ModelNew

    return ModelNew()


@lru_cache(maxsize=None)
def _dispatch_module():
    from sglang.srt.hardware_backend.npu.lplb.dispatch_probability_torch_reference_generated import (
        ModelNew,
    )

    return ModelNew()


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "npu",
) -> None:
    """Compile the IPM kernel for ``(nc, nv)`` before the first request.

    Mirrors ``cuda_solver.warmup``: the first launch pays the Triton-Ascend
    compilation cost, so do it once during solver construction.  Repeated
    calls for the same shape are skipped.
    """
    key = (int(nc), int(nv), int(num_iters), str(device))
    if key in _WARMED_UP:
        return

    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    _solve_module(num_iters)(A, b, c)

    _WARMED_UP.add(key)
    logger.info("LPLB Ascend NPU IPM solver: warmed up for (NC=%d, NV=%d)", nc, nv)


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Triton-Ascend interior-point solver.

    Args:
        A: Constraint matrix, shape ``(NC, NV)``, float32, on NPU.
        b: RHS vector, shape ``(NC,)``, float32, on NPU.
        c: Objective coefficients, shape ``(NV,)``, float32, on NPU.
        num_iters: Number of barrier iterations (default 5).
        result: Accepted for signature parity with the CUDA backend, but the
            generated kernel allocates its own output; the returned tensor is
            authoritative and ``result`` is not written.

    Returns:
        x: Solution vector, shape ``(NV,)``, float32.  The kernel writes 0.5
        for every entry on non-convergence.
    """
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"
    assert A.dtype == torch.float32
    return _solve_module(num_iters)(A, b, c)


def prep_lp_inputs(
    A_full: torch.Tensor,
    b: torch.Tensor,
    t1: torch.Tensor,
    global_counts: torch.Tensor,
    log_single: torch.Tensor,
    log_replicated: torch.Tensor,
    B1: torch.Tensor,
    A_base_row_sum: torch.Tensor,
) -> None:
    """Build the IPM inputs in place, matching ``cuda_solver.prep_lp_inputs``."""
    _prep_module()(
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
    )


def extract_log2phy_prob(
    log2phy_prob: torch.Tensor,
    x: torch.Tensor,
    t1: torch.Tensor,
    phy_single: torch.Tensor,
    phy_replicated: torch.Tensor,
    log2phy: torch.Tensor,
) -> None:
    """Turn the IPM output into ``log2phy_prob`` (in place).

    The generated ``lp_post`` wrapper takes the replicated-ratio prefix of the
    IPM solution rather than the full ``(NV,)`` vector, so slice it here.  Only
    the first ``NUM_RED_PHY`` entries of ``x`` are the replica ratios; the rest
    are the per-GPU slack variables, ``M`` and the Big-M column.
    """
    num_red_phy = phy_replicated.shape[0]
    _post_module()(
        log2phy_prob,
        x[:num_red_phy],
        t1,
        phy_single,
        phy_replicated,
        log2phy,
    )


def dispatch_probability(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample a physical expert per ``(token, slot)`` on Ascend NPU.

    Same contract as ``cuda_solver.dispatch_probability``: inverse-CDF sample
    over the per-row LP probabilities, falling back to a uniform choice over
    valid replicas when a row sums to zero.
    """
    original_shape = topk_ids.shape
    # ``topk_ids`` can be a non-contiguous column slice (e.g. routed columns of
    # a fused shared-expert layout), and the Triton kernel only sees the raw
    # pointer, so materialise a compact flat buffer first.
    flat_ids = topk_ids.reshape(-1).contiguous()
    n = flat_ids.numel()
    # Idle DP-attention ranks reach here with an empty tensor. The Triton launcher
    # would divide by the zero grid size, so short-circuit the no-op dispatch.
    if n == 0:
        return torch.empty(original_shape, dtype=topk_ids.dtype, device=topk_ids.device)

    if random_vals is None:
        random_vals = torch.rand(n, dtype=torch.float32, device=topk_ids.device)
    else:
        assert random_vals.shape == (n,), (
            f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"
        )
        random_vals = random_vals.contiguous()

    out = _dispatch_module()(flat_ids, log2phy_prob, log2phy_map, random_vals)
    return out.view(original_shape).to(topk_ids.dtype)
