"""Triton-Ascend implementation of the LPLB fused interior-point LP solver.

Phase-4 opt_iter_5 -- optimization point #6 (avoid scalar lowering) on top
of #7 (pass merge), #5 (scalar -> vector) and #1 (constexpr launch constants).

Inside the barrier loop the Cholesky factorisation and both triangular
substitutions are expressed as column sweeps.  Previously every step of the
3*NC-step solve chain needed a full-tile `tl.where` + `tl.sum` extraction and
then re-broadcast the resulting scalar; now each step costs one tile
reduction plus one 1-D scalar reduction and the rest of the work is a
vector FMA.  The factor is stored column-scaled (unit lower `M` plus the
1-D scale vector `s`), so no per-step diagonal extraction is required.
The Cholesky sweep and the forward substitution are fused, cutting the
serial step count of the whole solve from 3*NC to 2*NC.
Every ordering comparison is hoisted out of the barrier loop into a single
0/1 fp32 multiplier, so the inner loops contain only int32 EQ masks (which
vectorise on Ascend) and elementwise arithmetic.

Single fused kernel, one program, the whole barrier iteration lives inside
the kernel:

    x <- 1
    repeat num_iters:
        ax2   = A * x^2
        ax2a  = ax2 @ A^T + 1e-12 I
        ax2c  = ax2 @ c
        delta = solve(ax2a, ax2c)        # in-register Cholesky + fwd/bwd subst.
        r     = A^T delta
        d     = x * (c - r)
        d_max = max(d)
        alpha = 0.999 / d_max  (d_max > 1e-9) else 1.0
        x    *= 1 - alpha * d
    converged -> x, else 0.5 everywhere

Every shape enters the kernel as a constexpr (a dedicated binary per
(NC, NV)), the tail of the padded tiles is neutralised with masks and
zeros, and `A^T` is produced by a genuine transposed `tl.load` (never by
feeding `tl.trans` into `tl.dot`).
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ipm_kernel(
    A_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    NUM_ITERS: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    NCP: tl.constexpr,
    NVP: tl.constexpr,
):
    oc = tl.arange(0, NCP).to(tl.int32)
    ov = tl.arange(0, NVP).to(tl.int32)

    m_a = (oc[:, None] < NC) & (ov[None, :] < NV)
    m_at = (ov[:, None] < NV) & (oc[None, :] < NC)

    # A (NC, NV) row-major; AT is the same memory read with transposed indices so
    # that both tl.dot operands come from genuine loads (GC-15).
    a_t = tl.load(A_ptr + oc[:, None] * NV + ov[None, :], mask=m_a, other=0.0)
    at_t = tl.load(A_ptr + oc[None, :] * NV + ov[:, None], mask=m_at, other=0.0)
    b_t = tl.load(b_ptr + oc, mask=oc < NC, other=0.0)
    c_t = tl.load(c_ptr + ov, mask=ov < NV, other=0.0)

    eye = tl.where((oc[:, None] == oc[None, :]) & (oc[:, None] < NC), 1.0, 0.0)
    # OPT #6: the only ordering comparison in the whole kernel.  It is hoisted
    # out of the barrier loop and materialised as a 0/1 fp32 multiplier, so the
    # hot loops contain only int32 EQ masks (which vectorise) and multiplies --
    # int32 GT/GE/LE inside a loop get lowered to scalar loops on Ascend.
    lowm = tl.where(oc[:, None] >= oc[None, :], 1.0, 0.0)

    # Tail lanes of x stay 0 so they contribute nothing to ax2 / r / residual.
    x = tl.where(ov < NV, 1.0, 0.0)
    d_max = 0.0

    for _ in range(NUM_ITERS):
        x2 = x * x
        ax2 = a_t * x2[None, :]
        ax2a = tl.dot(ax2, at_t) + 1e-12 * eye
        ax2c = tl.sum(ax2 * c_t[None, :], axis=1)

        # ==================================================================
        # OPT #5 (scalar reduction -> vector broadcast/accumulate)
        #
        # The old form extracted every row/column of the tile with a full
        # `tl.where` + `tl.sum` pair and produced *scalars* that then had to be
        # broadcast again: 2 reductions per Cholesky step and 4 per
        # substitution step.  Here the factorisation is a column sweep so each
        # step costs exactly one tile reduction plus one 1-D scalar reduction.
        #
        #   A = L L^T ,  L = M * diag(s) ,  M unit lower (stored in `a`)
        #
        # `s[k] = L[k,k] = sqrt(Schur pivot)`, `a[i,k] = M[i,k] = L[i,k]/s[k]`.
        # ==================================================================
        # ==================================================================
        # OPT #7 (pass merge) + #5: the Cholesky factorisation and the
        # forward substitution are the *same* sweep over the same columns, so
        # they are fused into one loop.  At step k the column of M and the
        # final value of z[k] are both available, therefore the serial step
        # count of the whole solve drops from 3*NC to 2*NC.
        #
        #   A = L L^T ,  L = M * diag(s) ,  M unit lower (stored in `a`)
        #   z  solves  M z = ax2c
        # ==================================================================
        a = ax2a * lowm
        s = tl.zeros([NCP], dtype=tl.float32) + 1.0
        z = ax2c
        for k in tl.static_range(NC):
            col = tl.sum(tl.where(oc[None, :] == k, a, 0.0), axis=1)   # column k
            piv = tl.sum(tl.where(oc == k, col, 0.0))                  # scalar s_k
            piv = tl.sqrt(tl.maximum(piv, 1e-12))
            s = tl.where(oc == k, piv, s)
            m = tl.where(oc == k, 0.0, col / (piv * piv))
            a = tl.where(oc[None, :] == k, m[:, None], a)
            u = m * piv                                                # L[:, k]
            a = a - (u[:, None] * u[None, :]) * lowm
            zi = tl.sum(tl.where(oc == k, z, 0.0))                     # z[k]
            z = z - m * zi

        # ---- backward: M^T delta = z / s^2  (row sweep) ----
        v = z / (s * s)
        for i in tl.static_range(NC - 1, -1, -1):
            ri = tl.sum(tl.where(oc[:, None] == i, a, 0.0), axis=0)    # M[i, :]
            vi = tl.sum(tl.where(oc == i, v, 0.0))
            v = v - ri * vi
        delta = v


        r = tl.sum(a_t * delta[:, None], axis=0)
        d = x * (c_t - r)
        d_max = tl.max(tl.where(ov < NV, d, -3.0e38))
        alpha = tl.where(d_max > 1e-9, 0.999 / d_max, 1.0)
        x = x * (1.0 - alpha * d)

    # ---- convergence test (mirrors the reference exactly) ----
    ax = tl.sum(a_t * x[None, :], axis=1)
    resid = tl.max(tl.abs(ax - b_t))
    x_last = tl.sum(tl.where(ov == NV - 1, x, 0.0))
    ok = (d_max < 0.1) & (x_last >= 0.0) & (x_last < 1e-4) & (resid < 0.05)
    out_v = tl.where(ok, x, 0.5)
    tl.store(out_ptr + ov, out_v, mask=ov < NV)


def _ceil16(v: int) -> int:
    return max(16, (v + 15) // 16 * 16)


class ModelNew(nn.Module):
    def __init__(self, num_iters: int = 5):
        super(ModelNew, self).__init__()
        self.num_iters = int(num_iters)

    def forward(self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        b = b.contiguous()
        c = c.contiguous()
        nc, nv = A.shape
        ncp = _ceil16(nc)
        nvp = _ceil16(nv)
        out = torch.empty(nv, dtype=torch.float32, device=A.device)
        _ipm_kernel[(1,)](
            A,
            b,
            c,
            out,
            NUM_ITERS=self.num_iters,
            NC=nc,
            NV=nv,
            NCP=ncp,
            NVP=nvp,
        )
        return out
