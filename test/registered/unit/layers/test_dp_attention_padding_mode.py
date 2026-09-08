import unittest
from unittest import mock

from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDpAttentionPaddingMode(CustomTestCase):
    def _get_mode(
        self,
        global_num_tokens,
        *,
        backend="deepep_v2",
        deepep_mode=DeepEPMode.AUTO,
        is_extend_in_batch=False,
    ):
        with (
            mock.patch.object(dp_attention, "get_attention_dp_size", return_value=4),
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend(backend),
            ),
            mock.patch(
                "sglang.srt.layers.moe.utils.get_deepep_mode",
                return_value=deepep_mode,
            ),
            dp_attention.envs.SGLANG_DEEPEP_V2_FORCE_MAX_LEN.override(False),
        ):
            return dp_attention.DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=is_extend_in_batch,
                global_num_tokens=global_num_tokens,
            )

    def test_deepep_v2_idle_dp_uses_max_len(self):
        self.assertEqual(
            self._get_mode([8, 0, 0, 0]),
            dp_attention.DpPaddingMode.MAX_LEN,
        )

    def test_deepep_v2_nonzero_imbalanced_dp_uses_max_len(self):
        self.assertEqual(
            self._get_mode([8, 1, 1, 1]),
            dp_attention.DpPaddingMode.MAX_LEN,
        )

    def test_legacy_deepep_auto_decode_idle_dp_uses_max_len(self):
        self.assertEqual(
            self._get_mode([8, 0, 0, 0], backend="deepep"),
            dp_attention.DpPaddingMode.MAX_LEN,
        )

    def test_legacy_deepep_auto_decode_nonzero_imbalanced_dp_uses_max_len(self):
        self.assertEqual(
            self._get_mode([8, 1, 1, 1], backend="deepep"),
            dp_attention.DpPaddingMode.MAX_LEN,
        )

    def test_legacy_deepep_normal_decode_keeps_cost_heuristic(self):
        self.assertEqual(
            self._get_mode(
                [8, 0, 0, 0],
                backend="deepep",
                deepep_mode=DeepEPMode.NORMAL,
            ),
            dp_attention.DpPaddingMode.SUM_LEN,
        )


if __name__ == "__main__":
    unittest.main()
