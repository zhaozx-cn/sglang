"""Exercise the actual NPU execute override without launching device work."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "device_lengths,mode,raw_bs",
    [
        (True, ForwardMode.TARGET_VERIFY, 1),
        (True, ForwardMode.TARGET_VERIFY, 4),
        (False, ForwardMode.TARGET_VERIFY, 1),
        (False, ForwardMode.IDLE, 0),
    ],
)
@pytest.mark.parametrize("war_enabled", [False, True])
def test_actual_npu_execute_records_only_after_replay(
    device_lengths, mode, raw_bs, war_enabled
):
    order = []
    runner = NPUGraphRunner.__new__(NPUGraphRunner)
    # This fixture isolates the scheduler WAR event, not the independent
    # Target graph reuse guard normally initialized by the runner constructor.
    runner.target_graph_reuse_guard = False
    runner._target_graph_reuse_done = None
    old_event = object()
    runner.model_runner = SimpleNamespace(
        shared_read_done_event=old_event,
        is_draft_worker=device_lengths,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[])),
    )
    runner.bs = 16 if device_lengths else 2
    runner.raw_bs = raw_bs
    runner.raw_num_token = raw_bs * (7 if device_lengths else 8)
    runner.is_dllm = False
    runner.use_dspark_device_seq_lens = device_lengths
    runner.load_batch = Mock(side_effect=lambda *_: order.append("prepare"))
    runner._make_graph_key = lambda bs: bs
    runner._get_update_attr_name = lambda: "actual_seq_kvlen"
    runner._get_update_attr_type = lambda: []
    output = LogitsProcessorOutput(
        next_token_logits=torch.zeros(112, 3), hidden_states=torch.zeros(112, 2)
    )

    def replay(*args, **kwargs):
        assert runner.model_runner.shared_read_done_event is None
        order.append("replay")
        return output

    event = SimpleNamespace(record=lambda: order.append("record"))
    runner.device_module = SimpleNamespace(Event=lambda: event)
    runner.backend = SimpleNamespace(
        replay=Mock(side_effect=replay),
        replay_with_input_update=Mock(side_effect=replay),
    )
    batch = SimpleNamespace(
        needs_forward_metadata_init=lambda: True,
        forward_mode=mode,
        batch_size=raw_bs,
        original_global_num_tokens_cpu=[raw_bs, 0, 0, 0],
        seq_lens=torch.zeros(raw_bs, dtype=torch.int64),
        seq_lens_cpu=torch.full((raw_bs,), 128, dtype=torch.int64),
    )
    with (
        envs.SGLANG_ENABLE_WAR_BARRIER.override(war_enabled),
        envs.SGLANG_LOG_DECODE_GRAPH_KEY.override(True),
    ):
        result = runner.execute(batch)

    assert order == (
        ["prepare", "replay", "record"] if war_enabled else ["prepare", "replay"]
    )
    assert runner.model_runner.shared_read_done_event is (
        event if war_enabled else None
    )
    assert result.hidden_states.shape[0] == runner.raw_num_token
    if device_lengths:
        runner.backend.replay.assert_called_once()
        runner.backend.replay_with_input_update.assert_not_called()
    else:
        runner.backend.replay_with_input_update.assert_called_once()
        assert runner.backend.replay_with_input_update.call_args.kwargs["seq_lens"] == (
            [128] * raw_bs + [0] * (runner.bs - raw_bs)
        )


@pytest.mark.parametrize(
    "prefill,produced", [(False, False), (False, True), (True, False)]
)
def test_dspark_fallback_keeps_whole_worker_fence(monkeypatch, prefill, produced):
    from sglang.srt.speculative.dspark_components import dspark_worker_v2 as module

    monkeypatch.setattr(module, "is_npu", lambda: True)
    worker = module.DSparkWorkerV2.__new__(module.DSparkWorkerV2)
    target_event, draft_event = object(), object()
    worker.model_runner = SimpleNamespace(shared_read_done_event=target_event)
    worker.draft_model_runner = SimpleNamespace(shared_read_done_event=draft_event)
    worker.enable_draft_prefetch = True
    worker._verify_planner = Mock()
    worker._observers = Mock()
    expected = object()

    def forward(*args, **kwargs):
        if produced:
            worker._last_shared_read_runner = worker.draft_model_runner
        return expected

    worker._forward_decode = forward
    worker._forward_prefill = forward
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
        is_extend_in_batch=prefill,
    )
    assert worker.forward_batch_generation(batch) is expected
    if produced:
        assert worker.last_shared_read_runner is worker.draft_model_runner
        assert worker.draft_model_runner.shared_read_done_event is draft_event
    else:
        assert worker.last_shared_read_runner is worker.model_runner
        assert worker.model_runner.shared_read_done_event is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
