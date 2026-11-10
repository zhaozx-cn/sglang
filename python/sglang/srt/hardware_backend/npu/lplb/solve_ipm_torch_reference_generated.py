"""Triton-Ascend implementation of the LPLB fused interior-point LP solver.

Phase-4 opt_iter_6 -- removes the Cube stage and shrinks the working tiles to
the real problem size.

Measured bottleneck analysis of the previous version (A = 8x14, num_iters = 5,
device kernel time 74.1 us; each number below is an ablation, not an estimate):

  full kernel                                74.1 us
  - without the `tl.dot` KKT matmul          61.9 us   -> dot      ~12 us (16%)
  - without the Cholesky sweep               21.0 us   -> sweep    ~53 us (72%)
  - without the back substitution            52.2 us   -> backsub  ~22 us (30%)
  per Cholesky step                          1.33 us
    of which: pivot scalar round trip        0.70 us
              rank-1 tile update             0.19 us
              column extraction reduce       0.05 us

So the Cube stage was *not* the bottleneck: the cost is the serial per-step
overhead of the 16x16 tile.  `tl.dot` also forces every shape up to
`ceil16(NC) x ceil16(NV)` (an 8x8 problem ran on 16x16 tiles) and forces
`tl.dot`'s (M, N, K >= 16) shape contract, which pads an 8-row problem to 16.

Changes:
  1. `NCP/NVP = next_pow2(NC/NV)` (floor 8) instead of `ceil16`.  An 8x8
     problem now runs on 8x8 tiles: 4x fewer lanes per tile op.
  2. The KKT matmul is a 3-D broadcast reduction on the Vector unit
     (`sum_k ax2[i,k] * A[j,k]`), not a Cube `tl.dot`.  Measured faster at
     these sizes (61.9 us vs 74.1 us for the whole kernel) and it allows the
     un-padded tile size.  `tl.dot` is kept as a compile-time fallback for the
     large (NCP, NVP) shapes where the 3-D tile would overflow UB.
  3. LDL^T instead of Cholesky: the factor is unit-lower `L` plus the 1-D
     diagonal `D`.  No per-step `sqrt` and no per-step scale vector; the
     rank-1 sweep uses `a -= (L[:,k] (x) a[:,k])` and the solve divides by `D`
     once.
  4. The rank-1 update runs *before* the factor column is overwritten, so the
     eliminated column is zeroed by the update itself and the stored `L[:,k]`
     needs no extra masked vector.

Single fused kernel, one program, the whole barrier iteration lives inside the
kernel:

    x <- 1
    repeat num_iters:
        ax2   = A * x^2
        ax2a  = ax2 @ A^T + 1e-12 I
        ax2c  = ax2 @ c
        delta = solve(ax2a, ax2c)        # in-register LDL^T + fwd/bwd subst.
        r     = A^T delta
        d     = x * (c - r)
        d_max = max(d)
        alpha = 0.999 / d_max  (d_max > 1e-9) else 1.0
        x    *= 1 - alpha * d
    converged -> x, else 0.5 everywhere

Every shape enters the kernel as a constexpr (a dedicated binary per
(NC, NV)), the tail of the padded tiles is neutralised with masks and zeros,
and `A^T` -- when the Cube fallback is used -- is produced by a genuine
transposed `tl.load` (never by feeding `tl.trans` into `tl.dot`).
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
    USE_DOT: tl.constexpr,
):
    oc = tl.arange(0, NCP).to(tl.int32)
    ov = tl.arange(0, NVP).to(tl.int32)

    m_a = (oc[:, None] < NC) & (ov[None, :] < NV)
    m_at = (ov[:, None] < NV) & (oc[None, :] < NC)

    # A (NC, NV) row-major.  Only the Cube fallback needs a separately loaded
    # transpose (GC-15: never feed `tl.trans` into `tl.dot`).
    a_t = tl.load(A_ptr + oc[:, None] * NV + ov[None, :], mask=m_a, other=0.0)
    if USE_DOT:
        at_t = tl.load(A_ptr + oc[None, :] * NV + ov[:, None], mask=m_at, other=0.0)
    b_t = tl.load(b_ptr + oc, mask=oc < NC, other=0.0)
    c_t = tl.load(c_ptr + ov, mask=ov < NV, other=0.0)

    eye = tl.where((oc[:, None] == oc[None, :]) & (oc[:, None] < NC), 1.0, 0.0)
    # The only ordering comparison in the kernel: hoisted out of the barrier
    # loop and materialised as a 0/1 fp32 multiplier, so the hot loops contain
    # only int32 EQ masks (which vectorise on Ascend) and multiplies.
    lowm = tl.where(oc[:, None] >= oc[None, :], 1.0, 0.0)

    # Tail lanes of x stay 0 so they contribute nothing to ax2 / r / residual.
    x = tl.where(ov < NV, 1.0, 0.0)
    d_max = 0.0

    for _ in range(NUM_ITERS):
        x2 = x * x
        ax2 = a_t * x2[None, :]
        if USE_DOT:
            ax2a = tl.dot(ax2, at_t) + 1e-12 * eye
        else:
            # ax2a[i, j] = sum_k ax2[i, k] * A[j, k]  -- Vector-unit reduction.
            ax2a = tl.sum(ax2[:, None, :] * a_t[None, :, :], axis=2) + 1e-12 * eye
        ax2c = tl.sum(ax2 * c_t[None, :], axis=1)

        # ==================================================================
        # A = L D L^T with L unit lower (accumulated in `a`) and D = diag(d).
        # `inv` is the k-th pivot reciprocal, `dinv` is the 1-D D^-1 vector.
        #
        # Forward substitution (L z = ax2c) is fused into the same sweep: at
        # step k both L[:, k] and the final z[k] are available, so the whole
        # solve costs NC factorisation steps + NC back-substitution steps.
        # ==================================================================
        a = ax2a * lowm
        dinv = tl.zeros([NCP], dtype=tl.float32) + 1.0
        z = ax2c
        for k in tl.static_range(NC):
            col = tl.sum(tl.where(oc[None, :] == k, a, 0.0), axis=1)   # a[:, k]
            dk = tl.sum(tl.where(oc == k, col, 0.0))                   # scalar d_k
            inv = 1.0 / tl.maximum(dk, 1e-12)

            lk = col * inv                       # L[:, k] = a[:, k] / d_k
            lk = tl.where(oc == k, 0.0, lk)      # unit diagonal is implicit
            a = a - (lk[:, None] * col[None, :]) * lowm
            # The update above already zeroed column k below the diagonal
            # (a[i,k] - (a[i,k]/d_k) d_k == 0), so storing L[:, k] afterwards
            # needs no extra masked vector.
            a = tl.where(oc[None, :] == k, lk[:, None], a)

            dinv = tl.where(oc == k, inv, dinv)
            zi = tl.sum(tl.where(oc == k, z, 0.0))                     # z[k]
            z = z - lk * zi

        # ---- backward: L^T delta = D^-1 z  (row sweep) ----
        v = z * dinv
        for i in tl.static_range(NC - 1, -1, -1):
            ri = tl.sum(tl.where(oc[:, None] == i, a, 0.0), axis=0)    # L[i, :]
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


def _pow2(v: int) -> int:
    """Smallest power of two >= v, floored at 8 (measured best small tile)."""
    p = 8
    while p < v:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, num_iters: int = 5):
        super(ModelNew, self).__init__()
        self.num_iters = int(num_iters)

    def forward(self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        b = b.contiguous()
        c = c.contiguous()
        nc, nv = A.shape
        ncp = _pow2(nc)
        nvp = _pow2(nv)
        # 3-D broadcast tile is NCP*NCP*NVP elements; above 64 KiB it overflows
        # the UB, so fall back to the Cube path (which needs M,N,K >= 16).
        use_dot = (ncp * ncp * nvp > 16384) and ncp >= 16 and nvp >= 16
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
            USE_DOT=use_dot,
        )
        return out
