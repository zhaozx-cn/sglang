import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.models.kimi_k3 as kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_kimi_k3_topk_receives_expert_location_dispatch_info():
    expected_output = object()
    dispatch_info = object()
    owner = SimpleNamespace(
        layer_idx=17,
        topk=Mock(return_value=expected_output),
    )
    owner._expert_location_dispatch_info = lambda: (
        KimiK3MoE._expert_location_dispatch_info(owner)
    )
    hidden_states = torch.empty(2, 4)
    router_logits = torch.empty(2, 8)

    with patch.object(
        kimi_k3.ExpertLocationDispatchInfo,
        "init_new",
        return_value=dispatch_info,
    ) as init_new:
        actual = KimiK3MoE._select_experts(owner, hidden_states, router_logits)

    assert actual is expected_output
    init_new.assert_called_once_with(layer_id=17)
    owner.topk.assert_called_once_with(
        hidden_states,
        router_logits,
        expert_location_dispatch_info=dispatch_info,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
