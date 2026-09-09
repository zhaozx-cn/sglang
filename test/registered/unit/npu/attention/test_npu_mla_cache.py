"""CPU checks for logical prefix reads from ND and PA-NZ MLA cache pages."""

import unittest

import torch
from sglang.srt.hardware_backend.npu.attention.mla_cache import (
    assemble_mla_kv_from_prefix,
    gather_mla_cache_pages,
    gather_mla_cache_prefix,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMLACachePrefixRead(CustomTestCase):
    def test_page_order_and_layout(self):
        # Both latent and RoPE buffers must preserve every token's features.
        for page_size in (16, 128):
            for head_dim in (64, 512):
                logical = torch.arange(4 * page_size * head_dim).reshape(
                    4, page_size, 1, head_dim
                )
                packed = (
                    logical.reshape(4, page_size, head_dim // 16, 16)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape_as(logical)
                )
                for is_nz, cache in ((False, logical), (True, packed)):
                    for selected in ([3, 1, 3, 0], [], [2]):
                        with self.subTest(
                            page_size=page_size,
                            head_dim=head_dim,
                            is_nz=is_nz,
                            selected=selected,
                        ):
                            ids = torch.tensor(selected, dtype=torch.int32)
                            actual = gather_mla_cache_pages(cache, ids, is_nz=is_nz)
                            expected = logical[selected]
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertTrue(torch.equal(actual, expected))

    def test_partial_page_writes_preserve_prefix_features(self):
        # Populate NZ storage by scalar coordinates, independently of the
        # reshape/permute used by the reader. Leave unwritten slots sentinel-filled.
        page_size = 128
        slots = (128, 129, 255, 256, 383, 511)
        for head_dim in (64, 512):
            with self.subTest(head_dim=head_dim):
                tiles = head_dim // 16
                cache = torch.full((4, page_size, 1, head_dim), -1, dtype=torch.int64)
                logical = torch.full_like(cache, -1)
                flat = cache.view(-1)
                for slot in slots:
                    for dim in range(head_dim):
                        value = slot * 1000 + dim
                        logical[slot // page_size, slot % page_size, 0, dim] = value
                        offset = (
                            ((slot // page_size) * tiles + dim // 16) * page_size
                            + slot % page_size
                        ) * 16 + dim % 16
                        flat[offset] = value
                ids = torch.tensor([3, 1, 2, 1], dtype=torch.int64)
                actual = gather_mla_cache_pages(cache, ids, is_nz=True)
                self.assertTrue(torch.equal(actual, logical[[3, 1, 2, 1]]))
                # The former raw page gather must fail for this regression case.
                self.assertFalse(torch.equal(cache[ids], actual))

    def test_request_local_prefix_gather(self):
        page_size = 128
        head_dim = 64
        logical = torch.arange(6 * page_size * head_dim).reshape(
            6, page_size, 1, head_dim
        )
        packed = (
            logical.reshape(6, page_size, head_dim // 16, 16)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape_as(logical)
        )
        block_table = torch.tensor([4, 1, 5, 0], dtype=torch.int32)
        prefix_len = page_size * 2 + 17

        for is_nz, cache in ((False, logical), (True, packed)):
            with self.subTest(is_nz=is_nz):
                actual = gather_mla_cache_prefix(
                    cache, block_table, prefix_len, is_nz=is_nz
                )
                expected = logical[[4, 1, 5]].flatten(0, 1)[:prefix_len]
                self.assertEqual(actual.shape, (prefix_len, 1, head_dim))
                self.assertTrue(torch.equal(actual, expected))

    def test_request_local_prefix_gather_validates_capacity(self):
        cache = torch.zeros(2, 128, 1, 64)
        block_table = torch.tensor([0], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            gather_mla_cache_prefix(cache, block_table, -1, is_nz=False)
        with self.assertRaisesRegex(ValueError, "requires 2 blocks"):
            gather_mla_cache_prefix(cache, block_table, 129, is_nz=False)

    def test_assemble_mla_kv_matches_concat_reference(self):
        prefix_len, current_len, num_heads = 257, 17, 4
        nope_dim, rope_dim, value_dim = 128, 64, 128
        k_nope = torch.randn(prefix_len, num_heads, nope_dim)
        k_rope = torch.randn(prefix_len, 1, rope_dim)
        v_prefix = torch.randn(prefix_len, num_heads, value_dim)
        k_current = torch.randn(1, current_len, num_heads, nope_dim + rope_dim)
        v_current = torch.randn(1, current_len, num_heads, value_dim)

        actual_k, actual_v = assemble_mla_kv_from_prefix(
            k_nope, k_rope, v_prefix, k_current, v_current
        )
        expected_k = torch.cat(
            [
                torch.cat([k_nope, k_rope.expand(-1, num_heads, -1)], dim=-1)[None],
                k_current,
            ],
            dim=1,
        )
        expected_v = torch.cat([v_prefix[None], v_current], dim=1)

        self.assertTrue(actual_k.is_contiguous())
        self.assertTrue(actual_v.is_contiguous())
        self.assertTrue(torch.equal(actual_k, expected_k))
        self.assertTrue(torch.equal(actual_v, expected_v))


if __name__ == "__main__":
    unittest.main()
