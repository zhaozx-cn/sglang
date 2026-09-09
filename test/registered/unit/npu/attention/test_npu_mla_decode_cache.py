"""Check eager MLA dispatch and operator input/output contracts on CPU."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_mla_decode(operator, is_nz):
    root = Path(__file__).resolve().parents[5]
    attention = root / "python/sglang/srt/hardware_backend/npu/attention"
    source = attention / "ascend_backend.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward_decode"
    )
    branch = next(
        node
        for node in method.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not self.use_mla"
    )
    wrapper = ast.parse(
        "def call(self, q, q_rope, layer, forward_batch):\n"
        "    save_kv_cache = False\n"
    )
    wrapper.body[0].body.extend(branch.orelse)
    ast.fix_missing_locations(wrapper)
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(_npu_paged_attention_mla=operator),
        "is_fia_nz": lambda: is_nz,
        "next_power_of_2": lambda value: 1 << (value - 1).bit_length(),
    }
    reshape = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reshape_kv_for_fia_nz"
    )
    exec(
        compile(ast.Module(body=[reshape], type_ignores=[]), str(source), "exec"), scope
    )
    # Load the production layout helpers without importing the NPU runtime.
    helper = attention / "mla_cache.py"
    exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), scope)
    exec(compile(wrapper, str(source), "exec"), scope)
    return scope["call"]


def make_cache(page_size, head_dim, is_nz):
    logical = torch.arange(3 * page_size * head_dim, dtype=torch.float32).reshape(
        3, page_size, 1, head_dim
    )
    if not is_nz:
        return logical.clone(), logical
    # Populate physical NZ addresses independently of the reader's permutations.
    storage = torch.empty_like(logical)
    block = torch.arange(3)[:, None, None]
    token = torch.arange(page_size)[None, :, None]
    feature = torch.arange(head_dim)[None, None, :]
    offset = (
        (block * (head_dim // 16) + feature // 16) * page_size + token
    ) * 16 + feature % 16
    storage.view(-1)[offset.reshape(-1)] = logical.reshape(-1)
    return storage, logical


class TestEagerMLADecodeCache(unittest.TestCase):
    def test_explicit_paged_fallback_cache_contents(self):
        for page in (16, 128):
            for is_nz in (False, True):
                for heads in (3, 4, 6, 8):
                    use_fia = False
                    with self.subTest(page=page, nz=is_nz, heads=heads, fia=use_fia):
                        self._case(page, is_nz, heads, use_fia)

    def _case(self, page, is_nz, heads, use_fia):
        latent, logical_latent = make_cache(page, 512, is_nz)
        rope, logical_rope = make_cache(page, 64, is_nz)
        latent_before, rope_before = latent.clone(), rope.clone()
        seen = {}

        def paged(**kwargs):
            seen.update(kwargs)
            kwargs["out"].zero_()

        table = torch.tensor([[2, 1], [1, 2]], dtype=torch.int32)
        lengths = torch.tensor([page + 1, page * 2 - 1], dtype=torch.int32)
        owner = SimpleNamespace(
            use_fia=use_fia,
            graph_mode=False,
            page_size=page,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            forward_metadata=SimpleNamespace(
                block_tables=table, seq_lens_cpu_int=lengths
            ),
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: latent, get_value_buffer=lambda _: rope
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            tp_k_head_num=1,
            head_dim=576,
            scaling=576**-0.5,
        )
        # Include strided inputs, as the model's latent/RoPE projections use views.
        query = torch.randn(2, heads, 1024)[..., :512]
        query_rope = torch.randn(2, heads, 128)[..., :64]
        result = load_mla_decode(paged, is_nz)(
            owner, query, query_rope, layer, SimpleNamespace(batch_size=2)
        )
        expected = torch.cat([logical_latent, logical_rope], dim=-1)
        self.assertEqual(result.shape, (2, heads * 512))
        self.assertIs(seen["block_table"], table)
        self.assertIs(seen["context_lens"], lengths)
        self.assertEqual(seen["num_heads"], heads)
        self.assertTrue(seen["key_cache"].is_contiguous())
        torch.testing.assert_close(
            seen["query"], torch.cat([query, query_rope], dim=-1)
        )
        torch.testing.assert_close(latent, latent_before, rtol=0, atol=0)
        torch.testing.assert_close(rope, rope_before, rtol=0, atol=0)
        torch.testing.assert_close(seen["key_cache"], expected, rtol=0, atol=0)

    def test_fia_enabled_never_calls_atb_for_small_heads(self):
        for is_nz in (False, True):
            for heads in (1, 2, 3, 4, 6, 8):
                with self.subTest(nz=is_nz, heads=heads):
                    self._fia_case(is_nz, heads)

    def _fia_case(self, is_nz, heads):
        page = 128
        latent, _ = make_cache(page, 512, is_nz)
        rope, _ = make_cache(page, 64, is_nz)
        latent_before, rope_before = latent.clone(), rope.clone()
        table = torch.tensor([[2, 1], [1, 2]], dtype=torch.int32)
        lengths = torch.tensor([129, 253], dtype=torch.int32)
        owner = SimpleNamespace(
            use_fia=True,
            graph_mode=False,
            page_size=page,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            forward_metadata=SimpleNamespace(
                block_tables=table, seq_lens_cpu_int=lengths
            ),
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: latent, get_value_buffer=lambda _: rope
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            tp_k_head_num=1,
            head_dim=576,
            scaling=576**-0.5,
        )
        query = torch.randn(2, heads, 1024)[..., :512]
        query_rope = torch.randn(2, heads, 128)[..., :64]
        padded_heads = {1: 1, 2: 2, 3: 4, 4: 4, 6: 8, 8: 8}[heads]

        def reject_atb(**kwargs):
            self.fail("FIA-enabled MLA must not require ATB, including small heads")

        def fia(q, key, value, **kwargs):
            self.assertEqual(q.shape, (2, 1, padded_heads, 512))
            self.assertEqual(kwargs["query_rope"].shape, (2, 1, padded_heads, 64))
            self.assertEqual(kwargs["num_heads"], padded_heads)
            self.assertEqual(kwargs["num_key_value_heads"], 1)
            self.assertEqual(kwargs["input_layout"], "BSND")
            self.assertEqual(kwargs["block_size"], page)
            self.assertIs(kwargs["block_table"], table)
            self.assertIs(kwargs["actual_seq_lengths_kv"], lengths)
            self.assertIs(key, value)
            expected_shape = (3, 1, 32, page, 16) if is_nz else (3, page, 512)
            rope_shape = (3, 1, 4, page, 16) if is_nz else (3, page, 64)
            self.assertEqual(key.shape, expected_shape)
            self.assertEqual(kwargs["key_rope"].shape, rope_shape)
            torch.testing.assert_close(key.reshape(-1), latent_before.reshape(-1))
            torch.testing.assert_close(
                kwargs["key_rope"].reshape(-1), rope_before.reshape(-1)
            )
            torch.testing.assert_close(q[:, 0, :heads], query)
            torch.testing.assert_close(kwargs["query_rope"][:, 0, :heads], query_rope)
            self.assertEqual(torch.count_nonzero(q[:, :, heads:]).item(), 0)
            self.assertEqual(
                torch.count_nonzero(kwargs["query_rope"][:, :, heads:]).item(), 0
            )
            # Distinct output values exercise cropping of padded heads and flattening.
            return q + 1, None

        with patch.object(
            torch.ops.npu, "npu_fused_infer_attention_score", new=fia, create=True
        ):
            output = load_mla_decode(reject_atb, is_nz)(
                owner, query, query_rope, layer, SimpleNamespace(batch_size=2)
            )
        self.assertEqual(output.shape, (2, heads * 512))
        torch.testing.assert_close(output, (query + 1).reshape(2, -1))
        torch.testing.assert_close(latent, latent_before, rtol=0, atol=0)
        torch.testing.assert_close(rope, rope_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
