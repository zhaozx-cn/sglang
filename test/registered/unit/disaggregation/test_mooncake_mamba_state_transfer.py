"""Regression tests for Ascend KDA state transfer over Mooncake."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PREFILL_INDEX = 3
DECODE_INDEX = 10
ATTN_TP = 8

P_CONV_PTR = 0x1000_0000
P_TEMP_PTR = 0x2000_0000
D_CONV_PTR = 0x3000_0000
D_TEMP_PTR = 0x4000_0000

KDA_CHANNELS = 4608
P_CONV_ITEM = 3 * KDA_CHANNELS * 2
D_CONV_ITEM = 10 * KDA_CHANNELS * 2
KDA_CONV_GROUPS = [12288, 12288, 12288]

P_TEMP_ITEM = 12 * 128 * 128 * 2
D_TEMP_INCOMPATIBLE_ITEM = 12 * 128 * 128 * 4


class TestMooncakeMambaStateTransfer(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(MooncakeKVManager)
        self.manager.pp_size = 1
        self.manager.attn_tp_size = ATTN_TP
        self.manager._transfer_data = Mock(return_value=0)
        self.req = SimpleNamespace(mooncake_session_id="peer")

    def _send(self, *, include_incompatible_temporal=False):
        src_ptrs = [P_CONV_PTR]
        dst_ptrs = [D_CONV_PTR]
        src_lens = [P_CONV_ITEM]
        dst_lens = [D_CONV_ITEM]
        src_dims = [KDA_CHANNELS]
        dst_dims = [KDA_CHANNELS]
        conv_groups = [KDA_CONV_GROUPS]
        outer_counts = [3]

        if include_incompatible_temporal:
            src_ptrs.append(P_TEMP_PTR)
            dst_ptrs.append(D_TEMP_PTR)
            src_lens.append(P_TEMP_ITEM)
            dst_lens.append(D_TEMP_INCOMPATIBLE_ITEM)
            src_dims.append(12)
            dst_dims.append(12)
            conv_groups.append(None)
            outer_counts.append(1)

        return self.manager._send_mamba_state(
            self.req,
            [PREFILL_INDEX],
            src_ptrs,
            src_lens,
            src_dims,
            dst_ptrs,
            dst_lens,
            dst_dims,
            [DECODE_INDEX],
            conv_groups,
            outer_counts,
        )

    def test_expanded_decode_conv_uses_dst_stride_and_tail_alignment(self):
        self.assertEqual(self._send(), 0)

        expected_src = P_CONV_PTR + PREFILL_INDEX * P_CONV_ITEM
        expected_dst = (
            D_CONV_PTR + DECODE_INDEX * D_CONV_ITEM + D_CONV_ITEM - P_CONV_ITEM
        )
        self.manager._transfer_data.assert_called_once_with(
            "peer", [(expected_src, expected_dst, P_CONV_ITEM)]
        )
        self.assertEqual(
            expected_dst + P_CONV_ITEM,
            D_CONV_PTR + (DECODE_INDEX + 1) * D_CONV_ITEM,
        )

    def test_non_conv_length_mismatch_fails_before_transfer(self):
        with self.assertRaisesRegex(RuntimeError, r"outside the supported Ascend KDA"):
            self._send(include_incompatible_temporal=True)

        self.manager._transfer_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
