import types
import unittest

from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkExpertDistributionMetrics(CustomTestCase):
    def test_target_verify_preserves_expert_distribution_metrics(self):
        metrics = object()
        logits_output = object()
        target_output = types.SimpleNamespace(
            logits_output=logits_output,
            can_run_cuda_graph=True,
            expert_distribution_metrics=metrics,
        )
        target_worker = types.SimpleNamespace(
            forward_batch_generation=lambda **_kwargs: target_output
        )
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor.target_worker = target_worker

        verify_forward_batch = object()
        verify_input = types.SimpleNamespace(
            prepare_for_verify=lambda _batch, _worker: (verify_forward_batch, None)
        )
        batch = types.SimpleNamespace(seq_lens_cpu=None, seq_lens_sum=0)

        result = executor._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=None,
            seq_lens_sum_backup=0,
        )

        self.assertIs(result.logits_output, logits_output)
        self.assertTrue(result.can_run_cuda_graph)
        self.assertIs(result.expert_distribution_metrics, metrics)


if __name__ == "__main__":
    unittest.main()
