"""Test Qwen3-30B-A3B PD disaggregation GSM8K accuracy on NPU (Ascend).

NPU layout (8 NPUs total):
  Prefill server: TP=4, MOE_DP=2, EP=2, ATTN_CP=2 — NPUs 0–3
  Decode  server: TP=4, EP=2                        — NPUs 4–7 (base-gpu-id=4)

Within the prefill server, MOE_DP=2 creates 2 DP groups of TP=2 GPUs each.
ATTN_CP=2 splits the context across 2 CP workers within each DP group, so
each CP worker processes 1/2 of the prefill context (PCP path).
Decode does single-token autoregressive steps — moe_dp is not needed and
would trigger the assertion attn_cp_size == moe_dp_size (both default to 1).
PD disaggregation uses the Ascend transfer backend (no RDMA/IB devices required).
"""

import os
import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=600, suite="nightly-8-npu-a3", nightly=True)

QWEN3_MOE_MODEL = QWEN3_30B_A3B_WEIGHTS_PATH

# Prefill server: TP=4, MOE_DP=2, ATTN_CP=2 → 4 NPUs (0-3).
# Each DP group has 2 GPUs; each CP worker handles 1/2 of the context.
PREFILL_TP = 4
PREFILL_MOE_DP = 2
PREFILL_EP = 2
PREFILL_CP = 2
PREFILL_BASE_GPU_ID = 0

# Decode server uses the remaining 4 NPUs (4-7).
# No moe_dp — decode does single-token steps; moe_dp would require attn_cp to match.
DECODE_TP = 4
DECODE_EP = 2
DECODE_BASE_GPU_ID = PREFILL_TP  # = 4

# GSM8K accuracy floor — Qwen3-30B-A3B should comfortably exceed this.
GSM8K_MIN_ACCURACY = 0.75

# NPU-specific environment variables required for stable inference on Ascend.
_NPU_ENV_VARS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_USE_FIA": "1",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "HCCL_BUFFSIZE": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
    "HCCL_EXEC_TIMEOUT": "200",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENBLE_TORCH_COMILE": "1",
    "AUTO_USE_UC_MEMORY": "0",
    "P2P_HCCL_BUFFSIZE": "20",
}

_COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    "0.7",
]


class TestQwen3MoePDNpu(PDDisaggregationServerBase):
    """End-to-end GSM8K accuracy test for Qwen3-30B-A3B with PD disaggregation on NPU.

    Validates that:
    1. Prefill server starts with MOE_DP=2, ATTN_CP=2, and PCP enabled.
    2. KV cache is transferred to the decode server via the Ascend backend.
    3. GSM8K accuracy through the load balancer meets the minimum threshold.

    [Test Category] PD Disaggregation / Model Accuracy
    [Test Target] Qwen/Qwen3-30B-A3B
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Override transfer backend: NPU uses the Ascend backend — no RDMA/IB devices.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []

        cls.model = QWEN3_MOE_MODEL
        cls.npu_env = {**os.environ, **_NPU_ENV_VARS}

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            *_COMMON_SERVER_ARGS,
            "--disaggregation-mode",
            "prefill",
            "--max-running-requests",
            "1",
            "--tp",
            str(PREFILL_TP),
            "--moe-dp-size",
            str(PREFILL_MOE_DP),
            "--attn-cp-size",
            str(PREFILL_CP),
            "--enable-prefill-context-parallel",
            "--max-running-requests",
            "1",
            "--base-gpu-id",
            str(PREFILL_BASE_GPU_ID),
            *cls.transfer_backend,
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=cls.npu_env,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            *_COMMON_SERVER_ARGS,
            "--disaggregation-mode",
            "decode",
            "--tp",
            str(DECODE_TP),
            "--base-gpu-id",
            str(DECODE_BASE_GPU_ID),
            *cls.transfer_backend,
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=cls.npu_env,
        )

    def test_gsm8k_accuracy(self):
        """GSM8K accuracy validates PD disaggregation + PCP correctness on NPU end-to-end."""
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=32,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(
            f"GSM8K accuracy (NPU PD TP={PREFILL_TP} MOE_DP={PREFILL_MOE_DP} CP={PREFILL_CP}): "
            f"{metrics['accuracy']:.3f}"
        )
        self.assertGreater(metrics["accuracy"], GSM8K_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main(verbosity=3)
