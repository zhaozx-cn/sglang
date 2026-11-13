import torch
import triton
import triton.language as tl

_FALLBACK_VEC_CORES = 40
_BLOCK = 64

# (n, num_logical, max_copies) -> (grid, compiled, hooks, get_stream)
_NVEC = None

__all__ = ["dispatch", "dispatch_probability_kernel"]


def _num_vectorcore():
    """当前设备 AIV 核数(仅 host 侧元数据)。"""
    global _NVEC
    if _NVEC is None:
        try:
            from triton.runtime import driver

            v = int(driver.active.utils.get_device_properties(0)["num_vectorcore"])
            _NVEC = v if v > 0 else _FALLBACK_VEC_CORES
        except Exception:
            _NVEC = _FALLBACK_VEC_CORES
    return _NVEC


@triton.jit(do_not_specialize=["N", "NC"])
def dispatch_probability_kernel(
    topk_ptr,
    prob_ptr,
    map_ptr,
    rnd_ptr,
    out_ptr,
    N,
    NC,
    NL: tl.constexpr,
    MC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    # --- stage 0: 整表驻留 UB, 每 program 一次(被外层 block 循环摊销) --------
    # f 是 [NL*MC] 的线性下标, 转置后归约维(副本维)落在 axis 0。
    f = tl.arange(0, NL * MC)
    p = tl.trans(tl.reshape(tl.load(prob_ptr + f), (NL, MC)))
    mt = tl.trans(tl.reshape(tl.load(map_ptr + f), (NL, MC)).to(tl.int32))
    # 概率非负 => "行和 > 0" 与求和顺序无关, 可用树形归约判零。
    has_p = tl.sum(p, axis=0) > 0.0
    valid = (mt >= 0).to(tl.float32)
    # 零概率行回退为 "有效副本均匀采样": 扫描源换成 0/1 valid,
    # 一次 cumsum 同时覆盖 概率CDF / 有效副本前缀计数 两种语义。
    src = tl.where(has_p[None, :], p, valid)
    cum = tl.cumsum(src, axis=0)
    # 阈值取扫描的最后一个有效列 => 保持 row_sum == cum_last 不变量,
    # 避免 tl.sum 与 cumsum 结合顺序不同造成的 1ulp 边界翻转。
    tot = tl.max(cum, axis=0)
    cum_f = tl.reshape(cum, (NL * MC,))
    map_f = tl.reshape(mt, (NL * MC,))

    # --- stage 1: 逐 token block 只做 gather + 计数 ---------------------------
    c = tl.arange(0, MC)
    o = tl.arange(0, BLOCK)
    nb = tl.cdiv(N, BLOCK)
    for i in tl.range(0, tl.cdiv(nb, NC)):
        offs = (pid + i * NC) * BLOCK + o
        msk = offs < N
        lid = tl.load(topk_ptr + offs, mask=msk, other=0).to(tl.int32)
        # 表内 gather 没有 mask 参数: 把越界 lid 钳进 [0, NL-1] 保证索引合法
        # (合法输入 lid < NL, 该钳位不改变任何输出)
        lid = tl.minimum(tl.maximum(lid, 0), NL - 1)
        r0 = tl.load(rnd_ptr + offs, mask=msk, other=0.0)
        thr = r0 * tl.gather(tot, lid, 0)
        # 整行 [MC, BLOCK] 一次 gather, 再沿 axis 0 计数 <= thr 的列数
        ig = tl.reshape(c[:, None] * NL + lid[None, :], (MC * BLOCK,))
        row = tl.reshape(tl.gather(cum_f, ig, 0), (MC, BLOCK))
        chosen = tl.sum((row <= thr[None, :]).to(tl.int32), axis=0)
        chosen = tl.minimum(chosen, MC - 1)
        res = tl.gather(map_f, chosen * NL + lid, 0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, res, mask=msk)


class ModelNew(nn.Module):
    """Triton-Ascend port of ``dispatch_probability_torch_reference``."""

    def __init__(self):
        super().__init__()
        self._vec_cores = _num_vectorcore()

    def forward(
        self,
        topk_ids: torch.Tensor,
        log2phy_prob: torch.Tensor,
        log2phy_map: torch.Tensor,
        random_vals: torch.Tensor,
    ) -> torch.Tensor:
        n = topk_ids.numel()
        nl, mc = log2phy_prob.shape
        if n == 0:
            # 空 rank 可能拿到空张量; 0 网格非法
            return torch.empty_like(topk_ids)

        ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        prob = log2phy_prob.contiguous()
        mp = log2phy_map.contiguous()
        rv = random_vals.contiguous()
        # 内核按 flat row-major 写; ids 连续 => empty_like 连续且形状与输入一致。
        out = torch.empty_like(ids)

        g = min(_num_vectorcore(), (n + _BLOCK - 1) // _BLOCK)
        dispatch_probability_kernel[(g,)](
            ids, prob, mp, rv, out, n, g, nl, mc, _BLOCK
        )
        return out

