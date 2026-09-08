"""Equivalence tests for the EagleDraftWorker topk=1 chain fast path.

For topk=1 the draft tree degenerates to a chain, so `draft_forward` skips the
cat/topk/sort/gather of the slow path and returns pre-allocated constants. These
tests check that the pre-allocated `parent_list` / `top_scores_index` match the
slow path (`organize_draft_results`) for num_steps in {1, 2, 3, 4}.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_server_args(**fields):
    """server_args stand-in: carries fields and the override() entry point."""
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_chain_lists(num_steps: int, bs: int):
    """Build the (score, token, parents) lists a topk=1 chain produces.

    Shapes/values mirror `select_top_k_tokens` for topk=1: each step yields one
    token; the first step's parents are [-1, 0], later steps' parents are [i].
    """
    score_list, token_list, parents_list = [], [], []
    for i in range(num_steps):
        # Strictly decreasing scores, as a real chain produces (cumulative probs).
        score_list.append(torch.full((bs, 1, 1), float(num_steps - i), device=DEVICE))
        token_list.append(
            torch.arange(i * bs, (i + 1) * bs, device=DEVICE).unsqueeze(1)
        )
        if i == 0:
            parents_list.append(
                torch.tensor([-1, 0], dtype=torch.long, device=DEVICE).repeat(bs, 1)
            )
        else:
            parents_list.append(torch.full((bs, 1), i, dtype=torch.long, device=DEVICE))
    return score_list, token_list, parents_list


def _make_worker(num_steps: int, num_draft_tokens: int):
    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_draft_tokens
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=8)),
        max_running_requests=8,
    )
    return worker


def _make_backend_factory(decode_backend, draft_extend_backend, captured_kwargs=None):
    class FakeDraftBackendFactory:
        def __init__(self, *args, **kwargs):
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)

        def create_decode_backend(self):
            return decode_backend

        def create_draft_extend_backend(self):
            return draft_extend_backend

    return FakeDraftBackendFactory


class TestEagleWorkerV2Topk1FastPath(CustomTestCase):
    def setUp(self):
        # _rebuild_topk1_chain_buffers sizes its preallocation from the
        # published config: get_exec().graph.cuda_graph_config stays None on
        # the dummy-boundary publish (no resolution), so
        # get_schedule().max_running_requests alone sizes the buffers.
        override = get_context().override_server_args(max_running_requests=8)
        override.install()
        self.addCleanup(override.restore)

    def test_fast_path_matches_slow_path(self):
        bs = 3
        for num_steps in (1, 2, 3, 4):
            with self.subTest(num_steps=num_steps):
                num_draft_tokens = num_steps + 1
                worker = _make_worker(num_steps, num_draft_tokens)
                worker._rebuild_topk1_chain_buffers()

                score_list, token_list, parents_list = _make_chain_lists(num_steps, bs)
                ref_parent, ref_index, ref_tokens = organize_draft_results(
                    score_list, token_list, parents_list, num_draft_tokens
                )

                fast_parent = worker._topk1_parents_prealloc[:bs]
                fast_index = worker._topk1_score_indices_prealloc[:bs]
                fast_tokens = torch.cat(token_list, dim=1)

                self.assertEqual(fast_parent.shape, ref_parent.shape)
                self.assertEqual(fast_parent.tolist(), ref_parent.long().tolist())
                self.assertEqual(fast_index.tolist(), ref_index.long().tolist())
                self.assertEqual(fast_tokens.tolist(), ref_tokens.tolist())

                # The kernel reads these via data_ptr() as contiguous int64.
                self.assertEqual(fast_parent.dtype, torch.long)
                self.assertEqual(fast_index.dtype, torch.long)
                self.assertTrue(fast_parent.is_contiguous())
                self.assertTrue(fast_index.is_contiguous())

    def test_assert_on_inconsistent_steps_and_draft_tokens(self):
        # num_draft_tokens must equal num_steps + 1 for topk=1.
        worker = _make_worker(num_steps=3, num_draft_tokens=3)
        with self.assertRaises(AssertionError):
            worker._rebuild_topk1_chain_buffers()

    def test_idle_draft_runs_each_eager_forward_without_tree_layout(self):
        worker = object.__new__(EagleDraftWorker)
        worker.speculative_num_steps = 3
        worker.draft_attn_backend = SimpleNamespace(attn_backends=[object(), object()])
        worker.draft_runner = SimpleNamespace(
            canary_manager=None,
            forward=MagicMock(),
        )
        spec_info = SimpleNamespace(hidden_states=torch.empty((0, 8), device=DEVICE))
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            input_ids=torch.empty((0,), dtype=torch.long, device=DEVICE),
            out_cache_loc=torch.empty((0,), dtype=torch.long, device=DEVICE),
            spec_info=spec_info,
        )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.forward_context",
            side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
        ):
            result = worker.draft_forward(forward_batch)

        self.assertEqual(result, (None, None, None, None))
        self.assertEqual(worker.draft_runner.forward.call_count, 2)


class TestEagleDraftPrefetch(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_draft_prefetch=True,
            skip_draft_prefetch_seq_lens_cpu_sync=False,
        )
        override.install()
        self.addCleanup(override.restore)

    def test_prefill_seed_is_repeated_to_fixed_chain_width(self):
        worker = object.__new__(EagleDraftWorker)
        worker.enable_draft_prefetch = True
        worker.speculative_num_steps = 4
        worker.topk = 1
        worker.hot_token_id = None

        probabilities = torch.tensor([[0.8], [0.6]], device=DEVICE)
        indices = torch.tensor([[5], [7]], device=DEVICE)
        padded_p, padded_i = worker._pad_topk_for_draft_prefetch(probabilities, indices)

        self.assertTrue(torch.allclose(padded_p, probabilities.expand(-1, 4)))
        self.assertEqual(padded_i.tolist(), [[5] * 4, [7] * 4])

    def test_prefetched_chain_builds_verify_without_draft_forward(self):
        worker = object.__new__(EagleDraftWorker)
        worker.topk = 1
        worker.speculative_num_steps = 3
        worker.speculative_num_draft_tokens = 4
        worker.device = DEVICE
        worker.target_worker = object()
        worker.tree_mask_mode = object()
        draft_input = SimpleNamespace(
            topk_p=torch.ones((2, 3), device=DEVICE),
            topk_index=torch.tensor([[5, 50, 51], [6, 60, 61]], device=DEVICE),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=draft_input,
        )
        sentinel = object()

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_eagle_verify_input",
            return_value=sentinel,
        ) as build_verify:
            result = worker.prepare_verify_from_draft_prefetch(batch)

        self.assertIs(result, sentinel)
        parent_list = build_verify.call_args.args[2]
        selected_index = build_verify.call_args.args[3]
        draft_tokens = build_verify.call_args.args[4]
        self.assertEqual(parent_list.tolist(), [[-1, 0, 1], [-1, 0, 1]])
        self.assertEqual(selected_index.tolist(), [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(draft_tokens.tolist(), [[5, 50, 51], [6, 60, 61]])
        self.assertIsNone(build_verify.call_args.args[5])

    def test_draft_prefetch_concatenates_next_round_chain(self):
        worker = object.__new__(EagleDraftWorker)
        worker.req_to_token_pool = object()
        worker.cuda_graph_runner = SimpleNamespace(
            execute=MagicMock(
                return_value=(
                    None,
                    None,
                    torch.tensor([[5, 50, 51], [6, 60, 61]], device=DEVICE),
                    None,
                )
            )
        )
        worker.draft_runner = object()
        worker.draft_attn_backend = SimpleNamespace(needs_cpu_seq_lens=True)
        worker.topk = 1
        worker.speculative_num_steps = 3
        worker.hot_token_id = None

        original_spec = object()
        original_input = torch.tensor([1, 2], device=DEVICE)
        original_seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=DEVICE)
        original_seq_lens_cpu = original_seq_lens.cpu()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            seq_lens=original_seq_lens,
            seq_lens_cpu=original_seq_lens_cpu,
            seq_lens_sum=30,
            spec_info=original_spec,
            input_ids=original_input,
            device=DEVICE,
        )
        next_input = SimpleNamespace(
            topk_p=torch.ones((2, 1), device=DEVICE),
            topk_index=torch.tensor([[5], [6]], device=DEVICE),
        )
        result = SimpleNamespace(
            next_draft_input=next_input,
            new_seq_lens=torch.tensor([12, 23], dtype=torch.int32, device=DEVICE),
        )
        observed = {}

        def fake_prepare(*args):
            prepared_batch = args[2]
            observed["mode"] = prepared_batch.forward_mode
            observed["seq_lens"] = prepared_batch.seq_lens.clone()
            observed["seq_lens_cpu"] = prepared_batch.seq_lens_cpu.clone()
            return SimpleNamespace(), True

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
                side_effect=fake_prepare,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                return_value=contextlib.nullcontext(),
            ),
        ):
            worker.draft_prefetch(batch, result)

        self.assertEqual(observed["mode"], ForwardMode.DECODE)
        self.assertEqual(observed["seq_lens"].tolist(), [12, 23])
        self.assertEqual(observed["seq_lens_cpu"].tolist(), [12, 23])
        self.assertEqual(next_input.topk_index.tolist(), [[5, 50, 51], [6, 60, 61]])
        self.assertEqual(next_input.topk_p.tolist(), [[1.0] * 3, [1.0] * 3])
        self.assertIs(batch.spec_info, original_spec)
        self.assertIs(batch.input_ids, original_input)
        self.assertIs(batch.seq_lens, original_seq_lens)
        self.assertIs(batch.seq_lens_cpu, original_seq_lens_cpu)
        self.assertEqual(batch.seq_lens_sum, 30)


class TestEagleWorkerV2BackendFallback(CustomTestCase):
    def setUp(self):
        # The adaptive state-machine paths write live spec switches through
        # get_context().override, which needs a published config.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_missing_seed_cuda_graph_fallback(self):
        graph_result = (
            [],
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            None,
        )
        tree_result = (
            torch.empty((0,), dtype=torch.bool, device=DEVICE),
            torch.zeros((1,), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((2,), dtype=torch.long, device=DEVICE),
        )

        for seed_enabled, seed_present, expect_graph in (
            (True, False, False),
            (True, True, True),
            (False, False, True),
        ):
            with self.subTest(
                seed_enabled=seed_enabled,
                seed_present=seed_present,
            ):
                worker = object.__new__(EagleDraftWorker)
                worker.req_to_token_pool = None
                worker.cuda_graph_runner = SimpleNamespace(
                    execute=MagicMock(return_value=graph_result)
                )
                worker.draft_runner = SimpleNamespace(canary_manager=None)
                worker.topk = 1
                worker.speculative_num_steps = 1
                worker.speculative_num_draft_tokens = 2
                worker.device = DEVICE
                worker.tree_mask_mode = None
                worker.seed_dsa_topk_from_draft_extend = seed_enabled
                worker.index_share_for_mtp_iteration = True
                forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                worker.draft_forward = MagicMock(return_value=graph_result)
                attn_backend = SimpleNamespace(
                    verify_mask=None,
                    max_context_len=1,
                )
                worker.target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(attn_backend=attn_backend)
                )
                draft_input = SimpleNamespace(
                    bonus_tokens=torch.zeros((1,), dtype=torch.long, device=DEVICE),
                    dsa_topk_indices=(
                        torch.ones((1, 1), dtype=torch.int32, device=DEVICE)
                        if seed_present
                        else None
                    ),
                )
                batch = SimpleNamespace(
                    spec_info=draft_input,
                    forward_mode=ForwardMode.DECODE,
                    seq_lens_sum=1,
                    seq_lens=torch.ones((1,), dtype=torch.int32, device=DEVICE),
                )

                with patch(
                    "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
                    return_value=tree_result,
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
                    return_value=(forward_batch, True),
                ):
                    worker.draft(batch)

                self.assertEqual(worker.cuda_graph_runner.execute.called, expect_graph)
                self.assertEqual(worker.draft_forward.called, not expect_graph)

    def test_preserves_initialized_backend_when_draft_extend_backend_is_unset(self):
        worker = object.__new__(EagleDraftWorker)
        existing_backend = object()
        decode_backend = object()
        worker.server_args = _fake_server_args()
        worker.draft_runner = SimpleNamespace(attn_backend=existing_backend)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.seed_dsa_topk_from_draft_extend = False

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(decode_backend, None),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIsNone(worker.draft_extend_attn_backend)
        self.assertIs(worker.draft_runner.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_runner.attn_backend, existing_backend)

    def test_uses_draft_extend_backend_when_available(self):
        worker = object.__new__(EagleDraftWorker)
        existing_backend = object()
        decode_backend = object()
        draft_extend_backend = object()
        worker.server_args = _fake_server_args()
        worker.draft_runner = SimpleNamespace(attn_backend=existing_backend)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.seed_dsa_topk_from_draft_extend = True
        factory_kwargs = {}

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(
                decode_backend, draft_extend_backend, captured_kwargs=factory_kwargs
            ),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_extend_attn_backend, draft_extend_backend)
        self.assertIs(worker.draft_runner.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_runner.attn_backend, draft_extend_backend)
        self.assertTrue(factory_kwargs["seed_dsa_topk_from_draft_extend"])

    def _make_adaptive_worker(self, runner_attn_backend):
        """An EAGLEWorkerV2 with a draft worker whose state-machine fields are
        filled with sentinels, sufficient to drive _override_worker_state /
        apply_runtime_state without touching the GPU."""
        draft_runner = SimpleNamespace(
            draft_attn_backend=object(),
            attn_backend=runner_attn_backend,
        )
        draft_worker = SimpleNamespace(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=object(),
            draft_runner=draft_runner,
            # _override_worker_state / apply_runtime_state call this hook; the
            # topk=1 buffers are exercised by the fast-path tests above.
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = draft_worker
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(), decode_cuda_graph_runner=object()
            )
        )
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.server_args = _fake_server_args(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            cuda_graph_bs_decode=None,
            disable_cuda_graph=False,
        )
        return worker, draft_worker

    def test_override_worker_state_restores_runner_attn_backend(self):
        # build_adaptive_runtime_state runs init_attention_backend inside this
        # context for each candidate step; the runner backend it assigns must
        # not leak into the live worker.
        initial_backend = object()
        candidate_backend = object()
        worker, dw = self._make_adaptive_worker(initial_backend)

        with worker._override_worker_state(3, 4):
            dw.draft_runner.attn_backend = candidate_backend
            self.assertIs(dw.draft_runner.attn_backend, candidate_backend)

        self.assertIs(dw.draft_runner.attn_backend, initial_backend)

    def test_apply_runtime_state_updates_runner_attn_backend(self):
        # Switching to another step config must repoint the runner backend at
        # that config's draft-extend backend (read by the draft-extend forward).
        new_extend_backend = object()
        worker, dw = self._make_adaptive_worker(object())

        state = SpecRuntimeState(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            target_attn_backend=object(),
            target_graph_runner=object(),
            draft_extend_attn_backend=new_extend_backend,
            cuda_graph_runner_for_draft_extend=object(),
        )
        worker.apply_runtime_state(state)

        self.assertIs(dw.draft_runner.attn_backend, new_extend_backend)

    def test_spec_v2_attn_backends_include_draft_extend_fallback(self):
        target_backend = object()
        decode_backend = object()
        fallback_backend = object()

        worker = object.__new__(EAGLEWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker._draft_worker = SimpleNamespace(
            draft_attn_backend=decode_backend,
            draft_extend_attn_backend=None,
            draft_runner=SimpleNamespace(attn_backend=fallback_backend),
        )

        self.assertEqual(
            worker.spec_v2_attn_backends,
            (target_backend, decode_backend, fallback_backend),
        )


if __name__ == "__main__":
    unittest.main()
