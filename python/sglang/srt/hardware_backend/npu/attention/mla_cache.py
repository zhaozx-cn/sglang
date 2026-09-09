"""Read logical pages from the explicit PA-NZ storage of the NPU MLA cache."""

import torch


def gather_mla_cache_pages(
    cache: torch.Tensor, block_ids: torch.Tensor, *, is_nz: bool
) -> torch.Tensor:
    """Return selected pages in logical [blocks, page_size, 1, head_dim] order.

    NZ buffers retain that public shape, but their physical contents are
    [blocks, head_dim // 16, page_size, 16]. Restore token-major order before
    projecting cached latent vectors or concatenating their RoPE features.
    """
    pages = torch.index_select(cache, 0, block_ids)
    if not is_nz:
        return pages
    page_size, head_dim = cache.shape[1], cache.shape[-1]
    return (
        pages.view(block_ids.numel(), head_dim // 16, page_size, 16)
        .permute(0, 2, 1, 3)
        .reshape(block_ids.numel(), page_size, 1, head_dim)
    )


def gather_mla_cache_prefix(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    prefix_len: int,
    *,
    is_nz: bool,
) -> torch.Tensor:
    """Gather one request's prefix as logical ``[tokens, 1, head_dim]``.

    Keeping this operation request-local is important for long shared prefixes:
    gathering a flattened batch block table materializes the same physical cache
    pages once per request and makes peak memory scale with the sum of all prefix
    lengths.
    """
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")

    page_size = cache.shape[1]
    num_blocks = (prefix_len + page_size - 1) // page_size
    if num_blocks > block_table.numel():
        raise ValueError(
            f"Prefix of {prefix_len} tokens requires {num_blocks} blocks, "
            f"but block_table has only {block_table.numel()} entries"
        )

    pages = gather_mla_cache_pages(cache, block_table[:num_blocks], is_nz=is_nz)
    return pages.flatten(0, 1)[:prefix_len]


def assemble_mla_kv_from_prefix(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v_prefix: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build request-local contiguous FIA inputs with standard concatenation."""
    num_kv_heads = k_nope.shape[1]

    k_prefix = torch.cat(
        [k_nope, k_rope.expand(-1, num_kv_heads, -1)],
        dim=-1,
    )
    k_full = torch.cat([k_prefix[None], k_current], dim=1)
    v_full = torch.cat([v_prefix[None], v_current], dim=1)
    return k_full, v_full
