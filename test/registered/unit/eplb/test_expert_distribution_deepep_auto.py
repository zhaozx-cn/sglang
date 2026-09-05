from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.eplb.expert_distribution as expert_distribution
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _metadata():
    return SimpleNamespace(
        num_layers=2,
        num_physical_experts=4,
        num_local_physical_experts=2,
    )


def _make_auto_gatherer():
    with (
        get_context().override_server_args(
            expert_distribution_recorder_mode="stat",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            elastic_ep_backend=None,
        ),
        patch.object(expert_distribution, "get_device", return_value="cpu"),
    ):
        return expert_distribution._SinglePassGatherer.init_new(_metadata(), rank=1)


def test_deepep_auto_counts_selected_experts_for_extend():
    gatherer = _make_auto_gatherer()

    gatherer.reset()
    gatherer.on_select_experts(0, torch.tensor([[0, 3], [3, -1]]))
    result = gatherer.collect()["global_physical_count"]

    assert torch.equal(result, torch.tensor([[1, 0, 0, 2], [0, 0, 0, 0]]))


def test_deepep_auto_counts_selected_experts_for_decode():
    gatherer = _make_auto_gatherer()

    gatherer.reset()
    gatherer.on_select_experts(0, torch.tensor([[0, 3]]))
    # AUTO decode deliberately ignores this Python-side DeepEP hook: it is not
    # replayed by the NPU graph used by Kimi-K3.
    gatherer.on_deepep_dispatch_low_latency(0, torch.tensor([2, 4]))
    result = gatherer.collect()["global_physical_count"]

    assert torch.equal(result, torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]]))


def test_balancedness_ignores_layers_without_routed_tokens():
    counts = torch.tensor([[1, 3], [0, 0]])

    result = expert_distribution.compute_average_utilization_rate(counts)

    assert torch.isclose(result, torch.tensor(2 / 3), atol=1e-5)


def test_balancedness_marks_pass_without_routed_tokens_invalid():
    result = expert_distribution.compute_average_utilization_rate(
        torch.zeros((2, 4), dtype=torch.int32)
    )

    assert torch.isnan(result)
