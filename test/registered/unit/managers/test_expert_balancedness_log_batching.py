# ruff: noqa: E402

import types
import unittest
from collections import deque
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.eplb.expert_distribution import (
    EPLB_BALANCEDNESS_WINDOW_SIZES,
    ExpertDistributionMetrics,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
    _ExpertBalancednessDecodeLogWindow,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_reporter(*, decode_log_interval: int) -> SchedulerMetricsReporter:
    reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
    reporter.enable_metrics = False
    reporter.decode_log_interval = decode_log_interval
    reporter._eplb_balancedness_history = [
        deque(maxlen=window_size) for window_size in EPLB_BALANCEDNESS_WINDOW_SIZES
    ]
    reporter._eplb_decode_log_window = _ExpertBalancednessDecodeLogWindow()
    return reporter


def _make_result(
    forward_pass_id: int,
    balancedness: float,
    gpu_physical_count_sum: int,
) -> GenerationBatchResult:
    return GenerationBatchResult(
        expert_distribution_metrics=ExpertDistributionMetrics(
            forward_pass_id=forward_pass_id,
            eplb_balancedness=torch.tensor(balancedness),
            gpu_physical_count_sum=torch.tensor(gpu_physical_count_sum),
            reset_server_log_history=False,
        )
    )


class TestExpertBalancednessLogBatching(unittest.TestCase):
    @patch(
        "sglang.srt.managers.scheduler_components.metrics_reporter."
        "logs_expert_balancedness_to_server_log",
        return_value=True,
    )
    def test_decode_logs_one_aggregate_per_interval(self, _mock_logs_enabled):
        reporter = _make_reporter(decode_log_interval=3)
        batch = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)

        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.metrics_reporter",
            level="INFO",
        ) as captured:
            reporter.log_batch_result_stats(batch, _make_result(10, 0.1, 10))
            reporter.log_batch_result_stats(batch, _make_result(11, 0.2, 20))
            reporter.log_batch_result_stats(batch, _make_result(12, 0.3, 30))

        self.assertEqual(len(captured.output), 1)
        log_line = captured.output[0]
        self.assertIn("first_forward_pass_id=10", log_line)
        self.assertIn("forward_pass_id=12", log_line)
        self.assertIn("aggregated_forward_passes=3", log_line)
        self.assertIn("average_balancedness=0.200", log_line)
        self.assertIn("min_balancedness=0.100", log_line)
        self.assertIn("max_balancedness=0.300", log_line)
        self.assertIn("gpu_physical_count_sum=60", log_line)

    @patch(
        "sglang.srt.managers.scheduler_components.metrics_reporter."
        "logs_expert_balancedness_to_server_log",
        return_value=True,
    )
    def test_prefill_still_logs_each_forward(self, _mock_logs_enabled):
        reporter = _make_reporter(decode_log_interval=3)
        batch = types.SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.metrics_reporter",
            level="INFO",
        ) as captured:
            reporter.log_batch_result_stats(batch, _make_result(20, 0.4, 40))

        self.assertEqual(len(captured.output), 1)
        self.assertIn("forward_pass_id=20", captured.output[0])
        self.assertIn("current_pass_balancedness=0.400", captured.output[0])
        self.assertNotIn("aggregated_forward_passes", captured.output[0])


if __name__ == "__main__":
    unittest.main()
