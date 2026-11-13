import torch
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

    # A (NC, NV) row-major plus its transposed view.  `at_t` is materialised
    # once so that every A-side contraction can reduce along the contiguous
    # (axis=1) direction instead of a strided axis=0 sweep.
    a_t = tl.load(A_ptr + oc[:, None] * NV + ov[None, :], mask=m_a, other=0.0)
    at_t = tl.load(A_ptr + oc[None, :] * NV + ov[:, None], mask=m_at, other=0.0)
    b_t = tl.load(b_ptr + oc, mask=oc < NC, other=0.0)
    c_t = tl.load(c_ptr + ov, mask=ov < NV, other=0.0)

    eye = tl.where((oc[:, None] == oc[None, :]) & (oc[:, None] < NC), 1.0, 0.0)
    lowm = tl.where(oc[:, None] >= oc[None, :], 1.0, 0.0)

    # fp32 lane masks (0/1).  Multiplying by these is vector-unit work on
    # Ascend; `tl.where(<int compare>, <data tile>, 0.0)` scalarises the whole
    # select, so it is avoided everywhere on the hot path.
    xmask = tl.where(ov < NV, 1.0, 0.0)
    negbig = tl.where(ov < NV, 0.0, -3.0e38)
    lmask = tl.where(ov == NV - 1, 1.0, 0.0)

    x = xmask
    d_max = 0.0

    for _ in range(NUM_ITERS):
        x2 = x * x
        ax2 = a_t * x2[None, :]
        if USE_DOT:
            ax2a = tl.dot(ax2, at_t) + 1e-12 * eye
        else:
            ax2a = tl.sum(ax2[:, None, :] * a_t[None, :, :], axis=2) + 1e-12 * eye
        ax2c = tl.sum(ax2 * c_t[None, :], axis=1)

        a = ax2a * lowm
        dinv = tl.zeros([NCP], dtype=tl.float32) + 1.0
        z = ax2c
        for k in tl.static_range(NC):
            mcol = tl.where(oc[None, :] == k, 1.0, 0.0)     # [1, NCP]
            ncol = tl.where(oc[None, :] == k, 0.0, 1.0)     # [1, NCP]
            mrow = tl.where(oc == k, 1.0, 0.0)              # [NCP]
            nrow = tl.where(oc == k, 0.0, 1.0)              # [NCP]

            cmat = a * mcol                                 # only column k survives
            col = tl.sum(cmat, axis=1)                      # a[:, k]
            dk = tl.sum(col * mrow)                         # a[k, k]
            inv = 1.0 / tl.maximum(dk, 1e-12)

            lk = (col * inv) * nrow                         # unit diagonal
            # Column k is dropped exactly by subtracting `cmat`, and the two
            # remaining effects are folded into one rank-1 term:
            #   * columns j != k get the usual  a[:, j] - L[:, k] * a[k, j],
            #   * column k is rewritten to L[:, k] via tvec[k] == 1.
            tvec = mcol - col * ncol                        # [1, NCP]
            a = (a - cmat) + lk[:, None] * tvec

            dinv = dinv * nrow + inv * mrow
            zi = tl.sum(z * mrow)                           # z[k]
            z = z - lk * zi

        # ---- backward: L^T delta = D^-1 z  (row sweep) ----
        # The forward sweep no longer keeps the strictly upper triangle at zero
        # (it is never read there), so each extracted row is masked to j <= i.
        at = tl.trans(a)
        v = z * dinv
        for i in tl.static_range(NC - 1, -1, -1):
            mcol = tl.where(oc[None, :] == i, 1.0, 0.0)
            mrow = tl.where(oc == i, 1.0, 0.0)
            crow = tl.where(oc <= i, 1.0, 0.0)              # [NCP]
            ri = (tl.sum(at * mcol, axis=1)) * crow         # L[i, :]
            vi = tl.sum(v * mrow)
            v = v - ri * vi
        delta = v

        r = tl.sum(at_t * delta[None, :], axis=1)
        d = x * (c_t - r)
        d_max = tl.max(d * xmask + negbig)
        alpha = tl.where(d_max > 1e-9, 0.999 / d_max, 1.0)
        x = x * (1.0 - alpha * d)

    # ---- convergence test (mirrors the reference exactly) ----
    ax = tl.sum(a_t * x[None, :], axis=1)
    resid = tl.max(tl.abs(ax - b_t))
    x_last = tl.sum(x * lmask)
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
    def forward(self,A: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if not A.is_contiguous():
            A = A.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if not c.is_contiguous():
            c = c.contiguous()
        nc, nv = A.shape
        key = (A.device.index, A.dtype, nc, nv)
        ncp = _pow2(nc)
        nvp = _pow2(nv)
        # 3-D broadcast tile is NCP*NCP*NVP elements; above 64 KiB it overflows
        # the UB, so fall back to the Cube path (which needs M,N,K >= 16).
        use_dot = (ncp * ncp * nvp > 16384) and ncp >= 16 and nvp >= 16
        consts = (self.num_iters, nc, nv, ncp, nvp, use_dot)
        out = torch.empty(nv, dtype=torch.float32, device=A.device)
        _ipm_kernel[(1,)](A, b, c, out, *consts)
        return out

