import sys

import pytest
import torch

from sglang.srt.hardware_backend.npu import allocator_npu
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_alloc_extend_counts_pages_from_cpu_metadata(monkeypatch):
    allocator = object.__new__(allocator_npu.NPUPagedTokenToKVPoolAllocator)
    allocator.debug_mode = False
    allocator.page_size = 128
    allocator.roundup = 127
    allocator.need_sort = False
    allocator.device = "cpu"
    allocator.free_pages = torch.arange(300, dtype=torch.int64)

    prefix_lens_cpu = torch.tensor([127], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([129], dtype=torch.int64)
    calls = []

    def fake_get_num_new_pages(*, seq_lens, page_size, prefix_lens=None, **_):
        calls.append((seq_lens, page_size, prefix_lens))
        # Stay on the naive branch so this CPU-only unit test does not import
        # the NPU allocator kernel.
        return 256

    def fake_alloc_extend_naive(*args, **kwargs):
        return None

    monkeypatch.setattr(allocator_npu, "get_num_new_pages", fake_get_num_new_pages)
    monkeypatch.setattr(allocator_npu, "alloc_extend_naive", fake_alloc_extend_naive)

    result = allocator.alloc_extend(
        prefix_lens=torch.tensor([999], dtype=torch.int64),
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=torch.tensor([999], dtype=torch.int64),
        seq_lens_cpu=seq_lens_cpu,
        last_loc=torch.tensor([0], dtype=torch.int64),
        extend_num_tokens=1,
    )

    assert calls == [(seq_lens_cpu, 128, prefix_lens_cpu)]
    assert allocator.free_pages.numel() == 44
    assert result.shape == (1,)
    assert result.dtype == torch.int32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
