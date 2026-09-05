from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _make_utilization_accumulator(rank=0):
    accumulator = object.__new__(expert_distribution._StatAccumulator)
    accumulator._expert_location_metadata = SimpleNamespace(ep_size=2)
    accumulator._rank = rank
    accumulator._reset_server_log_history = True
    accumulator._history = MagicMock()
    accumulator._handle_metric_eplb_heatmap = MagicMock()
    return accumulator


def _utilization_patches(*, attn_dp_size, attn_tp_rank):
    return (
        patch.object(
            expert_distribution,
            "get_device_namespace",
            return_value=SimpleNamespace(device=torch.device("cpu")),
        ),
        patch.object(
            expert_distribution,
            "get_parallel",
            return_value=SimpleNamespace(
                attn_dp_size=attn_dp_size, attn_tp_rank=attn_tp_rank
            ),
        ),
        patch.object(
            expert_distribution,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(eplb_min_rebalancing_utilization_threshold=1.0)
            ),
        ),
        patch.object(
            expert_distribution,
            "logs_expert_balancedness_to_server_log",
            return_value=True,
        ),
        patch.object(torch.distributed, "reduce"),
    )


def test_dp_attention_balancedness_is_local_and_collective_free_on_leader():
    accumulator = _make_utilization_accumulator(rank=16)
    outputs = {}
    patches = _utilization_patches(attn_dp_size=4, attn_tp_rank=0)
    physical_count = torch.tensor([[3, 1, 0, 0], [1, 1, 2, 0]])
    physical_count_before = physical_count.clone()

    with patches[0], patches[1], patches[2], patches[3], patches[4] as reduce:
        accumulator._append_utilization_rate(7, physical_count, outputs)

    reduce.assert_not_called()
    assert torch.equal(physical_count, physical_count_before)
    assert outputs["metrics"].forward_pass_id == 7
    assert outputs["metrics"].gpu_physical_count_sum.item() == 8


def test_dp_attention_balancedness_is_silent_on_nonleader():
    accumulator = _make_utilization_accumulator(rank=17)
    outputs = {}
    patches = _utilization_patches(attn_dp_size=4, attn_tp_rank=1)

    with patches[0], patches[1], patches[2], patches[3], patches[4] as reduce:
        accumulator._append_utilization_rate(
            8, torch.tensor([[3, 1, 0, 0], [1, 1, 2, 0]]), outputs
        )

    reduce.assert_not_called()
    assert outputs == {}


def test_non_dp_balancedness_preserves_world_reduction():
    accumulator = _make_utilization_accumulator(rank=0)
    outputs = {}
    patches = _utilization_patches(attn_dp_size=1, attn_tp_rank=0)

    with patches[0], patches[1], patches[2], patches[3], patches[4] as reduce:
        accumulator._append_utilization_rate(
            9, torch.tensor([[3, 1, 0, 0], [1, 1, 2, 0]]), outputs
        )

    reduce.assert_called_once()
    assert outputs["metrics"].forward_pass_id == 9
