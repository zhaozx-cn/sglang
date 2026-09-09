"""Read logical pages from the explicit PA-NZ storage of the NPU MLA cache."""

import torch


def concat_mla_cache_for_paged_attention(
    latent: torch.Tensor, rope: torch.Tensor, *, is_nz: bool
) -> torch.Tensor:
    """Build token-major [blocks, page, 1, latent_dim + rope_dim] PA input.

    NZ buffers expose that public shape but store feature tiles before tokens.
    Concatenate their logical tile views so only the final combined cache is
    materialized, instead of allocating two full unpacked cache copies first.
    """
    if not is_nz:
        return torch.cat([latent, rope], dim=-1)
    blocks, page_size = latent.shape[:2]
    latent_tiles = latent.view(blocks, -1, page_size, 16).transpose(1, 2)
    rope_tiles = rope.view(blocks, -1, page_size, 16).transpose(1, 2)
    return torch.cat([latent_tiles, rope_tiles], dim=2).view(blocks, page_size, 1, -1)


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
