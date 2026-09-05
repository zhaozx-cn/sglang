from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_kimi_k3_text_exposes_expert_location_config():
    config = KimiLinearConfig(
        num_hidden_layers=61,
        num_experts=384,
        num_expert_group=8,
    )

    expert_config = KimiK3LinearForCausalLM.get_model_config_for_expert_location(config)

    assert expert_config.num_layers == 61
    assert expert_config.num_logical_experts == 384
    assert expert_config.num_groups == 8


def test_kimi_k3_multimodal_exposes_text_expert_location_config():
    config = KimiK3Config(
        text_config={
            "num_hidden_layers": 61,
            "num_experts": 384,
            "num_expert_group": 8,
        }
    )

    expert_config = KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
        config
    )

    assert expert_config.num_layers == 61
    assert expert_config.num_logical_experts == 384
    assert expert_config.num_groups == 8


def test_kimi_k3_dense_config_has_no_expert_location_config():
    config = KimiLinearConfig(num_experts=None)

    assert KimiK3LinearForCausalLM.get_model_config_for_expert_location(config) is None
