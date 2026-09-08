import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    should_defer_device_mlp_sync_metadata,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftProposal,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DEVICE = torch.device("cpu")


class TestDSparkDraftPrefetch(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=4,
            speculative_dspark_block_size=3,
            enable_draft_prefetch=True,
        )
        override.install()
        self.addCleanup(override.restore)

    def test_next_input_has_fixed_width_future_map_payload(self):
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7, 8], device=DEVICE),
            new_seq_lens=torch.tensor([11, 21], device=DEVICE),
        )

        self.assertEqual(tuple(draft_input.topk_p.shape), (2, 4))
        self.assertEqual(tuple(draft_input.topk_index.shape), (2, 4))
        self.assertEqual(draft_input.draft_prefetch_valid_cpu.tolist(), [False, False])
        self.assertFalse(draft_input.draft_prefetch_direct)
        payload = RelayPayload.from_draft_input(draft_input)
        self.assertIs(payload.topk_p, draft_input.topk_p)
        self.assertIs(payload.topk_index, draft_input.topk_index)
        self.assertTrue(SpeculativeAlgorithm.DSPARK.need_topk())

    def test_disabled_prefetch_schedule_has_no_relay_side_effects(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.enable_draft_prefetch = False
        next_input = DFlashDraftInputV2(
            topk_p=torch.ones((1, 1)),
            topk_index=torch.ones((1, 1), dtype=torch.int64),
            bonus_tokens=torch.tensor([7]),
            new_seq_lens=torch.tensor([11]),
            hidden_states=torch.empty((1, 0)),
        )

        produced = worker._schedule_draft_prefetch(
            batch=object(),
            next_draft_input=next_input,
            new_seq_lens=next_input.new_seq_lens,
            block_table_bound_cpu=None,
            target_model=object(),
            sampling_info=None,
        )

        self.assertFalse(produced)
        self.assertFalse(next_input.draft_prefetch_direct)

    def test_forward_resets_war_fence_to_target_before_idle_or_fallback(self):
        worker = object.__new__(DSparkWorkerV2)
        target_runner = SimpleNamespace(shared_read_done_event=None)
        draft_runner = object()
        worker.model_runner = target_runner
        worker.draft_model_runner = draft_runner
        worker.enable_draft_prefetch = True
        worker._last_shared_read_runner = draft_runner
        expected = object()
        worker._forward_decode = MagicMock(return_value=expected)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=MagicMock(return_value=False)),
            is_extend_in_batch=False,
        )

        result = worker.forward_batch_generation(batch)

        self.assertIs(result, expected)
        self.assertIs(worker.last_shared_read_runner, target_runner)

    def test_draft_reuse_fence_waits_only_at_next_draft_write(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.device = DEVICE
        read_done = object()
        worker.draft_model_runner = SimpleNamespace(shared_read_done_event=read_done)
        stream = MagicMock()
        device_module = SimpleNamespace(current_stream=MagicMock(return_value=stream))

        with patch("torch.get_device_module", return_value=device_module):
            worker._wait_for_previous_draft_read()

        stream.wait_event.assert_called_once_with(read_done)
        self.assertIsNone(worker.draft_model_runner.shared_read_done_event)

    def test_disabled_prefetch_uses_original_next_input_shape(self):
        override = get_context().override_server_args(enable_draft_prefetch=False)
        override.install()
        try:
            draft_input = make_next_draft_input(
                bonus_tokens=torch.tensor([7, 8], device=DEVICE),
                new_seq_lens=torch.tensor([11, 21], device=DEVICE),
            )
        finally:
            override.restore()

        self.assertEqual(tuple(draft_input.topk_p.shape), (2, 0))
        self.assertEqual(tuple(draft_input.topk_index.shape), (2, 0))
        self.assertFalse(draft_input.draft_prefetch_direct)
        self.assertIsNone(draft_input.draft_prefetch_valid_cpu)

    def test_num_token_non_padded_reuses_device_scalar(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._num_token_non_padded_buf = None

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft."
            "enable_num_token_non_padded",
            return_value=True,
        ):
            first = proposer._stage_num_token_non_padded(7, DEVICE)
            first_ptr = first.data_ptr()
            second = proposer._stage_num_token_non_padded(11, DEVICE)

        self.assertIs(first, second)
        self.assertEqual(second.data_ptr(), first_ptr)
        self.assertEqual(second.dtype, torch.int32)
        self.assertEqual(second.item(), 11)

    def test_target_graph_defers_throwaway_device_mlp_metadata(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([1, 2, 3, 4]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            out_cache_loc=torch.tensor([4, 5, 6, 7]),
            seq_lens_sum=8,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            can_run_decode_cuda_graph=True,
        )
        schedule_batch = SimpleNamespace(
            global_num_tokens=[4, 0, 0, 0],
            global_num_tokens_for_logprob=[4, 0, 0, 0],
            can_run_decode_cuda_graph=True,
        )

        with patch("sglang.srt.model_executor.forward_batch_info._is_npu", True):
            self.assertTrue(should_defer_device_mlp_sync_metadata(forward_batch))
            forward_batch.init_mlp_sync_metadata(schedule_batch, DEVICE)

        self.assertEqual(forward_batch.global_num_tokens_cpu, [4, 0, 0, 0])
        self.assertEqual(forward_batch.global_num_tokens_for_logprob_cpu, [4, 0, 0, 0])
        self.assertIsNone(forward_batch.global_num_tokens_gpu)
        self.assertIsNone(forward_batch.global_num_tokens_for_logprob_gpu)

    def test_target_cpu_lengths_replace_safe_allocation_bound(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=torch.tensor([1, 2, 3, 4]),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([17, 25]),
            out_cache_loc=torch.tensor([4, 5, 6, 7]),
            seq_lens_sum=64,
            seq_lens_cpu=torch.tensor([32, 32]),
            seq_lens_cpu_upper_bound=torch.tensor([32, 32]),
            deferred_seq_lens_cpu_resolver=lambda: (
                torch.tensor([21, 29]),
                50,
            ),
        )

        self.assertTrue(forward_batch.resolve_deferred_seq_lens_cpu())
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [21, 29])
        self.assertEqual(forward_batch.seq_lens_sum, 50)
        self.assertIsNone(forward_batch.deferred_seq_lens_cpu_resolver)

    def test_target_cpu_lengths_report_allocation_bound_miss(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([1, 2, 3, 4]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([31]),
            out_cache_loc=torch.tensor([4, 5, 6, 7]),
            seq_lens_sum=32,
            seq_lens_cpu=torch.tensor([32]),
            seq_lens_cpu_upper_bound=torch.tensor([32]),
            deferred_seq_lens_cpu_resolver=lambda: (torch.tensor([35]), 35),
        )

        self.assertFalse(forward_batch.resolve_deferred_seq_lens_cpu())
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [35])

    def test_deferred_cpu_lengths_use_captured_draft_input(self):
        """Target preparation replaces spec_info before its callback runs."""
        future_map = object.__new__(FutureMap)
        future_map.needs_cpu_seq_lens = True
        future_map.new_seq_lens_buf = torch.tensor([11, 22, 33], dtype=torch.int64)
        future_map.fwd_prepare_d2h_stream = None
        future_map.publish_ready = None

        producing_draft_input = SimpleNamespace(
            future_indices=torch.tensor([0, 2], dtype=torch.int64),
            draft_prefetch_seq_lens_cpu=None,
        )
        verify_input_without_future_indices = SimpleNamespace()
        batch = SimpleNamespace(
            spec_info=verify_input_without_future_indices,
            seq_lens=torch.tensor([11, 33], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0, 2], dtype=torch.int64),
        )

        future_map.resolve_seq_lens_cpu(
            batch,
            device_resolved=True,
            relay_input=producing_draft_input,
        )

        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 33])
        self.assertEqual(batch.seq_lens_sum, 44)

    def test_prefetched_cpu_lengths_wait_only_for_copy_event(self):
        future_map = object.__new__(FutureMap)
        future_map.needs_cpu_seq_lens = True
        future_map.prefetch_seq_lens_cpu_on_publish = True
        future_map.seq_lens_d2h_issued = 1
        future_map.new_seq_lens_buf = torch.tensor([11, 22, 33], dtype=torch.int64)
        future_map.new_seq_lens_cpu_pinned = torch.tensor(
            [11, 22, 33], dtype=torch.int64
        )
        future_map.seq_lens_d2h_copy_done = MagicMock()
        future_map.fwd_prepare_d2h_stream = MagicMock()
        future_map.publish_ready = MagicMock()

        producing_draft_input = SimpleNamespace(
            future_indices=torch.tensor([0, 2], dtype=torch.int64),
            draft_prefetch_seq_lens_cpu=None,
        )
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(),
            seq_lens=torch.tensor([11, 33], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0, 2], dtype=torch.int64),
        )

        future_map.resolve_seq_lens_cpu(
            batch,
            device_resolved=True,
            relay_input=producing_draft_input,
        )

        future_map.seq_lens_d2h_copy_done.synchronize.assert_called_once_with()
        future_map.fwd_prepare_d2h_stream.synchronize.assert_not_called()
        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 33])
        self.assertEqual(batch.seq_lens_sum, 44)

    def test_eager_fallback_materializes_deferred_mlp_metadata(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([1, 2, 3, 4]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            out_cache_loc=torch.tensor([4, 5, 6, 7]),
            seq_lens_sum=8,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            can_run_decode_cuda_graph=True,
            num_token_non_padded_cpu=4,
            global_num_tokens_cpu=[4, 0, 0, 0],
            global_num_tokens_for_logprob_cpu=[4, 0, 0, 0],
        )

        with (
            patch(
                "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.is_pin_memory_available",
                return_value=False,
            ),
        ):
            forward_batch.materialize_device_mlp_sync_metadata(DEVICE)

        self.assertEqual(int(forward_batch.num_token_non_padded), 4)
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [4, 0, 0, 0])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_gpu.tolist(), [4, 0, 0, 0]
        )

    def test_prefetched_block_rebuilds_proposal_and_confidence(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.enable_draft_prefetch = True
        worker._draft_prefetch_device_seq_lens = True
        worker._draft_prefetch_stats = {
            "produced": 0,
            "consumed": 0,
            "consume_miss": 0,
            "skipped_non_greedy": 0,
        }
        worker.verify_num_draft_tokens = 4
        worker.device = DEVICE
        worker._verify_planner = SimpleNamespace(carries_confidence=True)
        draft_input = DFlashDraftInputV2(
            topk_p=torch.tensor(
                [[1.0, 0.9, 0.8, 0.7], [1.0, 0.6, 0.5, 0.4]], device=DEVICE
            ),
            topk_index=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]], device=DEVICE
            ),
            bonus_tokens=torch.tensor([10, 20], device=DEVICE),
            new_seq_lens=torch.tensor([5, 6], device=DEVICE),
            hidden_states=torch.empty((2, 0), device=DEVICE),
            draft_prefetch_valid_cpu=torch.ones(2, dtype=torch.bool),
        )

        proposal = worker._proposal_from_draft_prefetch(
            draft_input=draft_input, sampling_info=None
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.draft_block_ids.tolist(), [[10], [20]])
        self.assertEqual(
            proposal.draft_block.draft_tokens.tolist(),
            [[11, 12, 13], [21, 22, 23]],
        )
        self.assertTrue(
            torch.allclose(
                proposal.confidence,
                torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]),
            )
        )
        self.assertEqual(worker._draft_prefetch_stats["consumed"], 1)

        draft_input.draft_prefetch_valid_cpu[1] = False
        self.assertIsNone(
            worker._proposal_from_draft_prefetch(
                draft_input=draft_input, sampling_info=None
            )
        )
        self.assertEqual(worker._draft_prefetch_stats["consume_miss"], 1)

    def test_prefetch_publishes_next_block_and_restores_batch(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.enable_draft_prefetch = True
        worker._draft_prefetch_device_seq_lens = True
        worker._draft_prefetch_stats = {
            "produced": 0,
            "consumed": 0,
            "consume_miss": 0,
            "skipped_non_greedy": 0,
        }
        worker.verify_num_draft_tokens = 4
        worker.device = DEVICE
        worker._block_pos_offsets = torch.arange(4, device=DEVICE)
        worker.model_runner = object()
        worker.draft_model_runner = object()
        worker._last_shared_read_runner = worker.model_runner
        worker._draft_context = MagicMock(return_value=contextlib.nullcontext())
        confidence = torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], device=DEVICE)
        proposal = DraftProposal(
            draft_block_ids=torch.tensor([[10], [20]], device=DEVICE),
            draft_block=DraftBlockResult(
                draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]], device=DEVICE),
                corrected_logits=None,
                greedy_mask=torch.ones(2, dtype=torch.bool, device=DEVICE),
                temperatures=torch.ones(2, device=DEVICE),
            ),
            draft_hidden=torch.empty((2, 3, 5), device=DEVICE),
        )
        worker._proposer = SimpleNamespace(propose=MagicMock(return_value=proposal))
        worker._verify_planner = SimpleNamespace(
            carries_confidence=True,
            compute_confidence_tensor=MagicMock(return_value=confidence),
        )

        original_spec = object()
        original_out_cache_loc = torch.tensor([3, 4], device=DEVICE)
        original_seq_lens = torch.tensor([10, 20], device=DEVICE)
        original_seq_lens_cpu = original_seq_lens.cpu()
        batch = SimpleNamespace(
            seq_lens=original_seq_lens,
            seq_lens_cpu=original_seq_lens_cpu,
            seq_lens_sum=30,
            spec_info=original_spec,
            out_cache_loc=original_out_cache_loc,
        )
        new_seq_lens = torch.tensor([12, 23], device=DEVICE)
        block_table_bound_cpu = torch.tensor([32, 48], dtype=torch.int32)
        next_input = make_next_draft_input(
            bonus_tokens=torch.tensor([10, 20], device=DEVICE),
            new_seq_lens=new_seq_lens,
        )
        observed = {}
        prefetched_positions = torch.tensor(
            [[12, 13, 14, 15], [23, 24, 25, 26]], device=DEVICE
        )
        prefetched_cache_loc_2d = torch.tensor(
            [[30, 31, 32, 33], [40, 41, 42, 43]], device=DEVICE
        )

        def fake_alloc(**kwargs):
            observed["seq_lens"] = kwargs["batch"].seq_lens.clone()
            observed["seq_lens_cpu"] = kwargs["batch"].seq_lens_cpu.clone()
            return VerifyWindow(
                positions_2d=prefetched_positions,
                verify_cache_loc=prefetched_cache_loc_2d.reshape(-1),
                verify_cache_loc_2d=prefetched_cache_loc_2d,
            )

        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.alloc_verify_window",
                side_effect=fake_alloc,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.spec_stage_span",
                return_value=contextlib.nullcontext(),
            ),
        ):
            produced = worker._schedule_draft_prefetch(
                batch=batch,
                next_draft_input=next_input,
                new_seq_lens=new_seq_lens,
                block_table_bound_cpu=block_table_bound_cpu,
                target_model=object(),
                sampling_info=None,
            )

        self.assertTrue(produced)
        self.assertTrue(next_input.draft_prefetch_direct)
        self.assertEqual(observed["seq_lens"].tolist(), [12, 23])
        self.assertEqual(observed["seq_lens_cpu"].tolist(), [32, 48])
        self.assertEqual(
            next_input.topk_index.tolist(),
            [[10, 11, 12, 13], [20, 21, 22, 23]],
        )
        self.assertTrue(torch.allclose(next_input.topk_p[:, 1:], confidence))
        self.assertEqual(next_input.draft_prefetch_valid_cpu.tolist(), [True, True])
        self.assertIsNone(next_input.draft_prefetch_seq_lens_cpu)
        self.assertIs(
            next_input.draft_prefetch_block_table_bound_cpu,
            block_table_bound_cpu,
        )
        reused_window = worker._verify_window_from_draft_prefetch(
            draft_input=next_input, bs=2
        )
        self.assertIs(reused_window.positions_2d, prefetched_positions)
        self.assertIs(reused_window.verify_cache_loc_2d, prefetched_cache_loc_2d)
        self.assertIs(batch.seq_lens, original_seq_lens)
        self.assertIs(batch.seq_lens_cpu, original_seq_lens_cpu)
        self.assertEqual(batch.seq_lens_sum, 30)
        self.assertIs(batch.spec_info, original_spec)
        self.assertIs(batch.out_cache_loc, original_out_cache_loc)
        self.assertEqual(worker._draft_prefetch_stats["produced"], 1)
        self.assertIs(worker.last_shared_read_runner, worker.draft_model_runner)

    def test_static_prefetch_does_not_launch_unused_probability_fills(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.enable_draft_prefetch = True
        worker._draft_prefetch_device_seq_lens = True
        worker._draft_prefetch_stats = {
            "produced": 0,
            "consumed": 0,
            "consume_miss": 0,
            "skipped_non_greedy": 0,
        }
        worker.verify_num_draft_tokens = 4
        worker.device = DEVICE
        worker._block_pos_offsets = torch.arange(4, device=DEVICE)
        worker.model_runner = object()
        worker.draft_model_runner = object()
        worker._last_shared_read_runner = worker.model_runner
        worker._draft_context = MagicMock(return_value=contextlib.nullcontext())
        proposal = DraftProposal(
            draft_block_ids=torch.tensor([[10], [20]], device=DEVICE),
            draft_block=DraftBlockResult(
                draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]], device=DEVICE),
                corrected_logits=None,
                greedy_mask=torch.ones(2, dtype=torch.bool, device=DEVICE),
                temperatures=torch.ones(2, device=DEVICE),
            ),
            draft_hidden=torch.empty((2, 3, 5), device=DEVICE),
        )
        worker._proposer = SimpleNamespace(propose=MagicMock(return_value=proposal))
        worker._verify_planner = SimpleNamespace(
            carries_confidence=False,
            compute_confidence_tensor=MagicMock(return_value=None),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], device=DEVICE),
            seq_lens_cpu=torch.tensor([10, 20]),
            seq_lens_sum=30,
            spec_info=object(),
            out_cache_loc=torch.tensor([3, 4], device=DEVICE),
        )
        next_input = make_next_draft_input(
            bonus_tokens=torch.tensor([10, 20], device=DEVICE),
            new_seq_lens=torch.tensor([12, 23], device=DEVICE),
        )
        next_input.topk_p.fill_(123.0)
        verify_window = VerifyWindow(
            positions_2d=torch.tensor(
                [[12, 13, 14, 15], [23, 24, 25, 26]], device=DEVICE
            ),
            verify_cache_loc=torch.tensor(
                [30, 31, 32, 33, 40, 41, 42, 43], device=DEVICE
            ),
            verify_cache_loc_2d=torch.tensor(
                [[30, 31, 32, 33], [40, 41, 42, 43]], device=DEVICE
            ),
        )

        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.alloc_verify_window",
                return_value=verify_window,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.spec_stage_span",
                return_value=contextlib.nullcontext(),
            ),
        ):
            produced = worker._schedule_draft_prefetch(
                batch=batch,
                next_draft_input=next_input,
                new_seq_lens=next_input.new_seq_lens,
                block_table_bound_cpu=torch.tensor([32, 48], dtype=torch.int32),
                target_model=object(),
                sampling_info=None,
            )

        self.assertTrue(produced)
        self.assertTrue(torch.all(next_input.topk_p == 123.0))
        self.assertEqual(
            next_input.topk_index.tolist(),
            [[10, 11, 12, 13], [20, 21, 22, 23]],
        )

    def test_overlap_filter_invalidates_direct_verify_window(self):
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7, 8], device=DEVICE),
            new_seq_lens=torch.tensor([11, 21], device=DEVICE),
        )
        draft_input.future_indices = torch.tensor([4, 5], device=DEVICE)
        # Filtering this process-local relay is valid only after the worker
        # successfully published a direct prefetched block.
        draft_input.draft_prefetch_direct = True
        draft_input.draft_prefetch_positions_2d = torch.tensor(
            [[11, 12, 13, 14], [21, 22, 23, 24]], device=DEVICE
        )
        draft_input.draft_prefetch_verify_cache_loc_2d = torch.tensor(
            [[31, 32, 33, 34], [41, 42, 43, 44]], device=DEVICE
        )

        draft_input.filter_batch(
            new_indices=torch.tensor([1], device=DEVICE), new_indices_cpu=[1]
        )

        self.assertIsNone(draft_input.draft_prefetch_positions_2d)
        self.assertIsNone(draft_input.draft_prefetch_verify_cache_loc_2d)
        self.assertEqual(draft_input.future_indices.tolist(), [5])
        self.assertEqual(tuple(draft_input.topk_p.shape), (1, 4))
        self.assertEqual(tuple(draft_input.topk_index.shape), (1, 4))

    def test_filter_invalidates_device_seq_allocation_bound(self):
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7, 8], device=DEVICE),
            new_seq_lens=torch.tensor([11, 21], device=DEVICE),
        )
        draft_input.future_indices = torch.tensor([4, 5], device=DEVICE)
        draft_input.draft_prefetch_block_table_bound_cpu = torch.tensor([32, 48])
        draft_input.draft_prefetch_valid_cpu.fill_(True)
        draft_input.filter_batch(
            new_indices=torch.tensor([1], device=DEVICE), new_indices_cpu=[1]
        )

        self.assertIsNone(draft_input.draft_prefetch_block_table_bound_cpu)
        self.assertEqual(draft_input.draft_prefetch_valid_cpu.tolist(), [False])

    def test_wait_and_plan_release_are_same_stream_noops(self):
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7], device=DEVICE),
            new_seq_lens=torch.tensor([11], device=DEVICE),
        )
        draft_input.wait_draft_prefetch()
        draft_input.release_draft_prefetch_plan()
        self.assertFalse(draft_input.draft_prefetch_cancelled)

    def test_discard_marks_same_stream_prefetch_cancelled(self):
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7], device=DEVICE),
            new_seq_lens=torch.tensor([11], device=DEVICE),
        )
        draft_input.discard_draft_prefetch()

        self.assertTrue(draft_input.draft_prefetch_cancelled)

    def test_decode_plan_waits_before_staging_host_lengths(self):
        """All device planning stays behind the caller-stream dependency."""
        order = []
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7], device=DEVICE),
            new_seq_lens=torch.tensor([11], device=DEVICE),
        )
        draft_input._prepare_batch_seq_lens_cpu_buf = torch.empty(1, dtype=torch.int64)
        draft_input._prepare_cur_kv_lens_cpu_buf = torch.empty(1, dtype=torch.int32)
        draft_input._prepare_nxt_kv_lens_cpu_buf = torch.empty(1, dtype=torch.int32)

        cur_device = MagicMock()
        nxt_device = MagicMock()
        cur_device.copy_.side_effect = lambda *args, **kwargs: order.append("cur_h2d")
        nxt_device.copy_.side_effect = lambda *args, **kwargs: order.append("nxt_h2d")
        cur_buf = MagicMock()
        nxt_buf = MagicMock()
        cur_buf.__getitem__.return_value = cur_device
        nxt_buf.__getitem__.return_value = nxt_device
        draft_input._prepare_cur_kv_lens_gpu_buf = cur_buf
        draft_input._prepare_nxt_kv_lens_gpu_buf = nxt_buf

        plan_stream = MagicMock()
        caller_stream = MagicMock()
        plan_stream.wait_stream.side_effect = lambda *args: order.append("plan_wait")
        caller_stream.wait_stream.side_effect = lambda *args: order.append(
            "caller_wait"
        )
        device_module = MagicMock()
        device_module.current_stream.return_value = caller_stream

        req = SimpleNamespace(
            kv=SimpleNamespace(kv_committed_len=10),
            sampling_params=SimpleNamespace(top_k=1),
            decode_batch_idx=0,
        )
        batch = SimpleNamespace(
            device=DEVICE,
            batch_size=MagicMock(return_value=1),
            maybe_evict_swa=MagicMock(),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            reqs=[req],
            tree_cache=object(),
            req_to_token_pool=object(),
            req_pool_indices=object(),
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

        with (
            patch.object(draft_input, "_ensure_prepare_length_buffers"),
            patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                return_value=(plan_stream, contextlib.nullcontext()),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.torch.get_device_module",
                return_value=device_module,
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.page_aligned_decode_alloc_lens",
                return_value=([10], [18], 8),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode",
                side_effect=lambda *args, **kwargs: order.append("alloc"),
            ),
        ):
            draft_input.prepare_for_decode(batch)

        self.assertEqual(
            order,
            ["plan_wait", "cur_h2d", "nxt_h2d", "alloc", "caller_wait"],
        )

    def test_decode_plan_stages_growth_lengths_with_original_copies(self):
        """Growth steps retain the proven allocator staging operation."""
        order = []
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7], device=DEVICE),
            new_seq_lens=torch.tensor([11], device=DEVICE),
        )
        draft_input._prepare_batch_seq_lens_cpu_buf = torch.empty(1, dtype=torch.int64)
        draft_input._prepare_cur_kv_lens_cpu_buf = torch.empty(1, dtype=torch.int32)
        draft_input._prepare_nxt_kv_lens_cpu_buf = torch.empty(1, dtype=torch.int32)

        cur_device = MagicMock()
        nxt_device = MagicMock()
        cur_device.copy_.side_effect = lambda *args, **kwargs: order.append("cur_h2d")
        nxt_device.copy_.side_effect = lambda *args, **kwargs: order.append("nxt_h2d")
        cur_buf = MagicMock()
        nxt_buf = MagicMock()
        cur_buf.__getitem__.return_value = cur_device
        nxt_buf.__getitem__.return_value = nxt_device
        draft_input._prepare_cur_kv_lens_gpu_buf = cur_buf
        draft_input._prepare_nxt_kv_lens_gpu_buf = nxt_buf

        req = SimpleNamespace(
            kv=SimpleNamespace(kv_committed_len=10),
            sampling_params=SimpleNamespace(top_k=1),
            decode_batch_idx=0,
        )
        batch = SimpleNamespace(
            device=DEVICE,
            batch_size=MagicMock(return_value=1),
            maybe_evict_swa=MagicMock(),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            reqs=[req],
            tree_cache=object(),
            req_to_token_pool=object(),
            req_pool_indices=object(),
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

        with (
            patch.object(draft_input, "_ensure_prepare_length_buffers"),
            patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                return_value=(None, contextlib.nullcontext()),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.page_aligned_decode_alloc_lens",
                return_value=([10], [18], 8),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode",
                side_effect=lambda *args, **kwargs: order.append("alloc"),
            ),
        ):
            draft_input.prepare_for_decode(batch)

        self.assertEqual(
            order,
            ["cur_h2d", "nxt_h2d", "alloc"],
        )
        cur_device.fill_.assert_not_called()
        nxt_device.fill_.assert_not_called()

    def test_decode_plan_preserves_zero_growth_bookkeeping_path(self):
        """A zero-growth step skips staging for every batch size."""
        order = []
        draft_input = make_next_draft_input(
            bonus_tokens=torch.tensor([7, 8], device=DEVICE),
            new_seq_lens=torch.tensor([11, 21], device=DEVICE),
        )
        draft_input._prepare_batch_seq_lens_cpu_buf = torch.empty(2, dtype=torch.int64)
        draft_input._prepare_cur_kv_lens_cpu_buf = torch.empty(2, dtype=torch.int32)
        draft_input._prepare_nxt_kv_lens_cpu_buf = torch.empty(2, dtype=torch.int32)
        cur_device = MagicMock()
        nxt_device = MagicMock()
        cur_buf = MagicMock()
        nxt_buf = MagicMock()
        cur_buf.__getitem__.return_value = cur_device
        nxt_buf.__getitem__.return_value = nxt_device
        draft_input._prepare_cur_kv_lens_gpu_buf = cur_buf
        draft_input._prepare_nxt_kv_lens_gpu_buf = nxt_buf
        reqs = [
            SimpleNamespace(
                kv=SimpleNamespace(kv_committed_len=10, kv_allocated_len=128),
                sampling_params=SimpleNamespace(top_k=1),
                decode_batch_idx=0,
            ),
            SimpleNamespace(
                kv=SimpleNamespace(kv_committed_len=20, kv_allocated_len=128),
                sampling_params=SimpleNamespace(top_k=1),
                decode_batch_idx=0,
            ),
        ]
        batch = SimpleNamespace(
            device=DEVICE,
            batch_size=MagicMock(return_value=2),
            maybe_evict_swa=MagicMock(),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=128),
            reqs=reqs,
            tree_cache=object(),
            req_to_token_pool=object(),
            req_pool_indices=object(),
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

        with (
            patch.object(draft_input, "_ensure_prepare_length_buffers"),
            patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                side_effect=lambda *_: (
                    order.append("get_plan_stream") or (None, contextlib.nullcontext())
                ),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.page_aligned_decode_alloc_lens",
                return_value=([128, 128], [128, 128], 0),
            ),
            patch(
                "sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode",
                side_effect=lambda *args, **kwargs: order.append("alloc"),
            ),
        ):
            draft_input.prepare_for_decode(batch)

        self.assertEqual(order, ["get_plan_stream", "alloc"])
        cur_device.copy_.assert_not_called()
        nxt_device.copy_.assert_not_called()
        cur_device.fill_.assert_not_called()
        nxt_device.fill_.assert_not_called()
        self.assertEqual(draft_input.nxt_kv_lens_cpu.tolist(), [128, 128])
        self.assertEqual([req.decode_batch_idx for req in reqs], [1, 1])


if __name__ == "__main__":
    unittest.main()
