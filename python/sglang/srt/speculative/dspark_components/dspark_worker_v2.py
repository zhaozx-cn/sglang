import logging
from contextlib import nullcontext
from dataclasses import replace
from typing import Optional

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_diagnostics import (
    configure_diagnostics,
    diagnostic_stage,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftProposal,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    maybe_build_draft_sampler,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    VerifyWindow,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_tp_context,
    prepare_mamba_track_for_verify,
    spec_stage_span,
)
from sglang.srt.utils import (
    is_cuda,
    is_cuda_alike,
    is_npu,
    is_pin_memory_available,
)

logger = logging.getLogger(__name__)


class DSparkWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
        draft_worker_cls: type[TpModelWorker] = TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = get_schedule().page_size
        self.device = target_worker.device

        self._draft_is_moe = draft_is_deepseek_v4()
        self._draft_dp_context_enabled = (
            get_parallel().enable_dp_attention and not self._draft_is_moe
        )
        self._is_pd_prefill = get_disagg().disaggregation_mode == "prefill"
        self._decode_graph_allowed = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self._is_pd_prefill
        )
        if (
            get_parallel().enable_dp_attention
            and self._draft_is_moe
            and ps.attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        with self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                ps=replace(ps, pp_rank=0, pp_size=1),
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                attention_backend_override=(
                    DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                ),
                draft_worker_cls=draft_worker_cls,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None
        # Updated for every forward according to the phase that actually ran
        # last.  This cannot be derived from the feature flag: idle ranks and
        # fallback rounds do not execute the post-verify draft prefetch.
        self._last_shared_read_runner = self.model_runner

        configure_diagnostics(self.device, tp_rank=ps.tp_rank, dp_rank=ps.attn_dp_rank)

        # The mask token is input-only (it is embedded, never sampled), so its
        # bound is the embedding-table row count: the PADDED vocab when the
        # target pads its embedding (e.g. Inkling true vocab 200058, padded
        # 201024, mask 200064), else the plain vocab size.
        target_model_config = self.target_worker.model_runner.model_config
        target_embed_rows = (
            getattr(target_model_config.hf_text_config, "padded_vocab_size", None)
            or target_model_config.vocab_size
        )
        # muP targets declare logits_mup_width_multiplier; the draft was
        # trained against the folded head, so compute_base_logits divides.
        self.draft_model.logits_mup_width_multiplier = getattr(
            target_model_config.hf_text_config, "logits_mup_width_multiplier", None
        )
        self._target_is_mambaish = mambaish_config(target_model_config) is not None
        runtime_config = resolve_runtime_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=int(target_embed_rows),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.sample_from_anchor = bool(self.draft_model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id
        self.enable_draft_prefetch = bool(get_spec().enable_draft_prefetch)
        self._draft_prefetch_stats = {
            "produced": 0,
            "consumed": 0,
            "consume_miss": 0,
            "skipped_non_greedy": 0,
            "target_prep_deferred": 0,
            "target_prep_sync_fallback": 0,
        }
        self._draft_prefetch_greedy_mask = None
        self._draft_prefetch_temperatures = None
        # The Ascend DSPark draft backend consumes device seq_lens directly.
        # Prefetch therefore stays on the forward stream: no blocking D2H,
        # Future.result, cross-stream event, or Gloo/plan handoff is needed.
        self._draft_prefetch_async = False
        # Attention backends are initialized in a later startup phase.
        self._draft_prefetch_device_seq_lens = False

        parallel = get_parallel()
        self._tp_sync = SpecTpSync(
            parallel.attn_tp_group
            if parallel.enable_dp_attention
            else parallel.tp_group
        )
        self._draft_graph_group = (
            parallel.attn_tp_group
            if self._draft_dp_context_enabled
            else parallel.tp_group
        )

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, query_token_num=%s, "
                "sample_from_anchor=%s, mask_token_id=%s, markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self.query_token_num,
                self.sample_from_anchor,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.query_token_num), device=self.device
        )

        if getattr(self.draft_model, "uses_own_vocab_modules", False):
            if self.ps.tp_rank == 0:
                logger.info(
                    "DSpark draft uses its checkpoint-local embedding and LM head."
                )
        else:
            target_model = self.target_worker.model_runner.model
            lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError(
                    "DSpark requires the target model to expose `lm_head` with `weight`."
                )
            self.draft_model.attach_shared_modules(
                embed_tokens=unwrap_lora_layer(
                    self._resolve_target_embed_tokens(target_model)
                ),
                lm_head=lm_head,
            )

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=self.ps.tp_rank,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_sync=self._tp_sync,
        )
        if (
            get_parallel().enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
        )
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            tp_sync=self._tp_sync,
            dp_moe_sync=self._draft_is_moe and get_parallel().enable_dp_attention,
        )
        self._verify_epilogue = None
        if (
            self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
            and is_cuda()
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                tp_sync=self._tp_sync,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            tp_sync=self._tp_sync,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

        self._forced_budget_frac: Optional[float] = None
        self._need_mamba_verify_commit = False

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.ps.tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

        if self._is_pd_prefill and not self._draft_is_moe:
            self.draft_model.prune_to_ctx_kv_injection()

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._verify_planner.carries_confidence

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    @property
    def last_shared_read_runner(self):
        # The scheduler's next allocation can mutate the shared req_to_token
        # table. A successful prefetched Draft ACLGraph replay reads that table
        # after Target has finished, so its POST_REPLAY event—not Target's
        # earlier event—is the only sound WAR boundary.  Keeping only a narrow
        # draft-input reuse fence is insufficient: req_to_token would still be
        # read and written concurrently and can feed an invalid DDR address to
        # cache_loc/FIA kernels.
        return self._last_shared_read_runner

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _draft_context(self):
        if self._draft_dp_context_enabled:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    @diagnostic_stage("draft_buffer_wait")
    def _wait_for_previous_draft_read(self) -> None:
        """Secondary guard for direct/non-scheduler draft-buffer reuse.

        The overlap scheduler normally consumes this event through
        ``last_shared_read_runner`` before it mutates shared allocation state.
        Retain this local guard for fallback/idle paths and direct worker calls
        that can reach a new proposal without passing that scheduler boundary.
        """
        read_done = getattr(self.draft_model_runner, "shared_read_done_event", None)
        if read_done is None:
            return
        self.draft_model_runner.shared_read_done_event = None
        torch.get_device_module(self.device).current_stream().wait_event(read_done)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        with self._draft_context():
            self._draft_worker.init_attention_backends()
        self._draft_prefetch_device_seq_lens = is_npu() and bool(
            getattr(
                self.draft_model_runner.attn_backend,
                "use_dspark_device_verify",
                False,
            )
        )
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = self._decode_graph_allowed
        available_mem = self._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM,
            self.device,
            self.gpu_id,
            group=self._draft_graph_group,
        )
        if is_cuda_alike() and capture_decode_cuda_graph:
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                # Keep the draft model graph enabled when folded proposal is
                # disabled, but do not capture the proposal head as a tail
                # hook. The proposer will compute base logits and the Markov
                # block eagerly from the graph's hidden states instead. Apart
                # from being the intended precision fallback, skipping the
                # unused hook avoids paying for two proposal computations.
                if envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get():
                    self._draft_sampler = self._maybe_build_draft_sampler(
                        available_memory_gb=available_mem
                    )
                    if self._draft_sampler is not None:
                        self.draft_model_runner.capture_tail_hooks.append(
                            make_draft_sampler_capture_hook(self._draft_sampler)
                        )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )
        # Device sequence lengths are valid only when the draft runner really
        # captured a device-length ACLGraph. If graph capture was disabled for low
        # memory, retain the functional host-metadata prefetch fallback.
        draft_graph_runner = self.draft_model_runner.decode_cuda_graph_runner
        self._draft_prefetch_device_seq_lens = bool(
            is_npu()
            and draft_graph_runner is not None
            and getattr(draft_graph_runner, "use_dspark_device_seq_lens", False)
        )

    def _maybe_build_draft_sampler(self, *, available_memory_gb: float):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.ps.tp_rank,
            tp_sync=self._tp_sync,
            available_memory_gb=available_memory_gb,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        dumped = self._observers.dump_info_records() or {}
        dumped["draft_prefetch"] = {
            "enabled": self.enable_draft_prefetch,
            "async_npu": self._draft_prefetch_async,
            "same_stream": True,
            "device_seq_lens": self._draft_prefetch_device_seq_lens,
            "deferred_graph_metadata_h2d": bool(
                self.enable_draft_prefetch
                and self._draft_prefetch_device_seq_lens
                and envs.SGLANG_DSPARK_DEFER_TARGET_METADATA.get()
            ),
            **self._draft_prefetch_stats,
        }
        return dumped

    def clear_info_records(self) -> None:
        self._observers.clear_info_records()
        for key in self._draft_prefetch_stats:
            self._draft_prefetch_stats[key] = 0

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    @diagnostic_stage("worker_forward", device=True, iteration=True)
    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        seq_lens_cpu_resolver=None,
    ) -> GenerationBatchResult:
        # Safe default for prefill, idle decode, and every prefetch fallback.
        # _schedule_draft_prefetch switches this only after it really enqueues
        # a post-verify draft graph.
        self._last_shared_read_runner = self.model_runner
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            result = self._forward_prefill(batch, on_publish)
        else:
            result = self._forward_decode(
                batch,
                on_publish,
                grammar_barrier,
                seq_lens_cpu_resolver=seq_lens_cpu_resolver,
            )
        if (
            is_npu()
            and self.enable_draft_prefetch
            and self._last_shared_read_runner is self.model_runner
        ):
            # A target-graph event does not cover the worker's later KV
            # injection/accept-commit reads. Without a successful prefetch,
            # retain the scheduler's whole-forward fence for these paths.
            self.model_runner.shared_read_done_event = None
        return result

    @diagnostic_stage("target_prefill_and_inject", device=True)
    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if get_parallel().enable_dp_attention:
                self.target_worker.forward_batch_generation(
                    batch, capture_hidden_mode=CaptureHiddenMode.FULL
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch_output = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = next_token_ids.device
        pin_memory = is_pin_memory_available(device)
        ctx_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        # unified_kv injects into the SWA ring keyed by (draft req slot, position);
        # thread the per-token state_slot + the req's final position so the
        # injector keeps only the last SWA window (older prefill tokens share a
        # ring slot and would race). Cheap; only consumed under unified_kv.
        state_slot = final_pos = None
        if is_unified_kv_triton():
            repeats = ctx_lens.to(torch.int64)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64), repeats
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64), repeats
            )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
            state_slot=state_slot,
            final_pos=final_pos,
        )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and get_parallel().enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _proposal_from_draft_prefetch(
        self,
        *,
        draft_input: DFlashDraftInputV2,
        sampling_info,
    ) -> Optional[DraftProposal]:
        """Consume the fixed-width DSpark block relayed from the prior round."""
        if not self.enable_draft_prefetch:
            return None
        if sampling_info is not None and not sampling_info.is_all_greedy:
            self._draft_prefetch_stats["consume_miss"] += 1
            return None
        if (
            draft_input.topk_p.ndim != 2
            or draft_input.topk_index.ndim != 2
            or draft_input.topk_p.shape[1] != self.verify_num_draft_tokens
            or draft_input.topk_index.shape[1] != self.verify_num_draft_tokens
        ):
            self._draft_prefetch_stats["consume_miss"] += 1
            return None
        # Prefill and any mixed batch row without a prefetched block carry a
        # false host-side validity bit. Fall back for the whole batch so all
        # TP/DP ranks make the same draft-forward decision. Do not inspect the
        # device topk tensor with .item(): that would serialize every decode.
        valid_cpu = draft_input.draft_prefetch_valid_cpu
        if valid_cpu is None or not bool(torch.all(valid_cpu)):
            self._draft_prefetch_stats["consume_miss"] += 1
            return None

        verify_ids_2d = draft_input.topk_index
        bs = int(verify_ids_2d.shape[0])
        draft_tokens = verify_ids_2d[:, 1:]
        confidence = (
            draft_input.topk_p[:, 1:]
            if self._verify_planner.carries_confidence
            else None
        )
        greedy_mask = self.__dict__.get("_draft_prefetch_greedy_mask")
        temperatures = self.__dict__.get("_draft_prefetch_temperatures")
        if greedy_mask is None or greedy_mask.numel() < bs:
            capacity = max(
                bs, 32, 0 if greedy_mask is None else greedy_mask.numel() * 2
            )
            greedy_mask = torch.ones(capacity, dtype=torch.bool, device=self.device)
            temperatures = torch.ones(capacity, dtype=torch.float32, device=self.device)
            self._draft_prefetch_greedy_mask = greedy_mask
            self._draft_prefetch_temperatures = temperatures
        proposal = DraftProposal(
            draft_block_ids=verify_ids_2d[:, :1],
            draft_block=DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=None,
                greedy_mask=greedy_mask[:bs],
                temperatures=temperatures[:bs],
            ),
            draft_hidden=None,
            confidence=confidence,
            confidence_tap=None,
            # The prefetched token block is correct for graph and eager verify,
            # but folded accept owns separate capture-time buffers. Keep accept
            # eager until those buffers are explicitly relayed as well.
            folded=False,
        )
        self._draft_prefetch_stats["consumed"] += 1
        return proposal

    def _verify_window_from_draft_prefetch(
        self, *, draft_input: DFlashDraftInputV2, bs: int
    ) -> Optional[VerifyWindow]:
        positions_2d = draft_input.draft_prefetch_positions_2d
        cache_loc_2d = draft_input.draft_prefetch_verify_cache_loc_2d
        expected = (bs, self.verify_num_draft_tokens)
        if (
            positions_2d is None
            or cache_loc_2d is None
            or tuple(positions_2d.shape) != expected
            or tuple(cache_loc_2d.shape) != expected
        ):
            return None
        return VerifyWindow(
            positions_2d=positions_2d,
            verify_cache_loc=cache_loc_2d.reshape(-1),
            verify_cache_loc_2d=cache_loc_2d,
        )

    def _draft_prefetch_next(
        self,
        *,
        batch: ScheduleBatch,
        next_draft_input: DFlashDraftInputV2,
        new_seq_lens: torch.Tensor,
        block_table_bound_cpu: Optional[torch.Tensor],
        target_model,
        sampling_info,
    ) -> bool:
        """Pre-run the next DSpark proposal and publish it in next_draft_input."""
        if not self.enable_draft_prefetch:
            return False
        # Draft sampling needs corrected logits/probabilities in the next
        # accept step. The first functional path prefetches greedy requests and
        # leaves sampling requests on the existing proposal path.
        if sampling_info is not None and not sampling_info.is_all_greedy:
            self._draft_prefetch_stats["skipped_non_greedy"] += 1
            return False

        original_seq_lens = batch.seq_lens
        original_seq_lens_cpu = batch.seq_lens_cpu
        original_seq_lens_sum = batch.seq_lens_sum
        original_spec_info = batch.spec_info
        original_out_cache_loc = batch.out_cache_loc
        try:
            if next_draft_input.draft_prefetch_cancelled:
                return False
            if self._draft_prefetch_device_seq_lens and block_table_bound_cpu is None:
                # Bootstrap/mixed batches without scheduler over-allocation do
                # not have a safe device-only window yet. Fall back next round
                # instead of introducing a D2H on the critical path.
                return False
            batch.seq_lens = new_seq_lens
            if self._draft_prefetch_device_seq_lens:
                batch.seq_lens_cpu = block_table_bound_cpu
                # The CPU tensor is an allocation bound, not an exact logical
                # length. Keep the sum absent so no downstream path mistakes it
                # for sum(device seq_lens).
                batch.seq_lens_sum = None
                next_draft_input.draft_prefetch_block_table_bound_cpu = (
                    block_table_bound_cpu
                )
            else:
                batch.seq_lens_cpu = new_seq_lens.to("cpu")
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                next_draft_input.draft_prefetch_seq_lens_cpu = batch.seq_lens_cpu
            batch.spec_info = next_draft_input

            bs = int(new_seq_lens.numel())
            verify_window = alloc_verify_window(
                batch=batch,
                bs=bs,
                device=self.device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
                model_runner=self.model_runner,
            )
            with self._draft_context(), spec_stage_span("draft_prefetch"):
                proposal = self._proposer.propose(
                    batch=batch,
                    draft_input=next_draft_input,
                    verify_window=verify_window,
                    bs=bs,
                    device=self.device,
                    target_model=target_model,
                    sampling_info=sampling_info,
                )

            confidence = proposal.confidence
            if confidence is None:
                confidence = self._verify_planner.compute_confidence_tensor(
                    draft_hidden=proposal.draft_hidden,
                    anchor_tokens=proposal.draft_block_ids[:, 0],
                    draft_tokens=proposal.draft_block.draft_tokens,
                    confidence_tap=proposal.confidence_tap,
                )

            draft_block_ids = proposal.draft_block_ids[:, :1]
            draft_tokens = proposal.draft_block.draft_tokens
            expected_shape = (
                draft_block_ids.shape[0],
                draft_block_ids.shape[1] + draft_tokens.shape[1],
            )
            if expected_shape != tuple(next_draft_input.topk_index.shape):
                raise RuntimeError(
                    "DSpark draft-prefetch block shape mismatch: "
                    f"proposal={expected_shape}, "
                    f"relay={tuple(next_draft_input.topk_index.shape)}."
                )
            next_draft_input.topk_index[:, :1].copy_(draft_block_ids)
            next_draft_input.topk_index[:, 1:].copy_(draft_tokens)
            if self._verify_planner.carries_confidence:
                # Direct-prefetch consumption only reads columns 1..gamma as
                # confidence; column 0 belongs to the legacy FutureMap payload
                # convention and is never relayed/read on this path. Leave it
                # untouched to avoid one device Fill between Draft and Verify.
                if confidence is not None:
                    if confidence.shape != next_draft_input.topk_p[:, 1:].shape:
                        raise RuntimeError(
                            "DSpark draft-prefetch confidence shape mismatch: "
                            f"confidence={tuple(confidence.shape)}, "
                            f"relay={tuple(next_draft_input.topk_p[:, 1:].shape)}."
                        )
                    next_draft_input.topk_p[:, 1:].copy_(confidence)
                elif next_draft_input.topk_p.shape[1] > 1:
                    # A confidence-enabled fallback still needs deterministic
                    # relay contents even though the K3 path normally supplies
                    # confidence above.
                    next_draft_input.topk_p[:, 1:].zero_()
            next_draft_input.draft_prefetch_positions_2d = verify_window.positions_2d
            next_draft_input.draft_prefetch_verify_cache_loc_2d = (
                verify_window.verify_cache_loc_2d
            )
            if next_draft_input.draft_prefetch_valid_cpu is None:
                next_draft_input.draft_prefetch_valid_cpu = torch.ones(
                    draft_block_ids.shape[0], dtype=torch.bool
                )
            else:
                next_draft_input.draft_prefetch_valid_cpu.fill_(True)
            self._draft_prefetch_stats["produced"] += 1
            return True
        finally:
            batch.seq_lens = original_seq_lens
            batch.seq_lens_cpu = original_seq_lens_cpu
            batch.seq_lens_sum = original_seq_lens_sum
            batch.spec_info = original_spec_info
            batch.out_cache_loc = original_out_cache_loc

    @diagnostic_stage("draft_prefetch", device=True)
    def _schedule_draft_prefetch(
        self,
        *,
        batch: ScheduleBatch,
        next_draft_input: DFlashDraftInputV2,
        new_seq_lens: torch.Tensor,
        block_table_bound_cpu: Optional[torch.Tensor],
        target_model,
        sampling_info,
    ) -> bool:
        if not self.enable_draft_prefetch:
            return False
        next_draft_input.draft_prefetch_cancelled = False
        self._wait_for_previous_draft_read()
        produced = self._draft_prefetch_next(
            batch=batch,
            next_draft_input=next_draft_input,
            new_seq_lens=new_seq_lens,
            block_table_bound_cpu=block_table_bound_cpu,
            target_model=target_model,
            sampling_info=sampling_info,
        )
        next_draft_input.draft_prefetch_direct = produced
        if produced:
            # NPUGraphRunner publishes the sound POST_REPLAY event on the
            # draft runner.  The next scheduler iteration must consume that
            # event before modifying the shared req_to_token allocation map.
            self._last_shared_read_runner = self.draft_model_runner
        return produced

    def _forward_decode(
        self,
        batch: ScheduleBatch,
        on_publish,
        grammar_barrier=None,
        seq_lens_cpu_resolver=None,
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            if get_parallel().enable_dp_attention:
                if self._draft_is_moe:
                    self._wait_for_previous_draft_read()
                    self._proposer.run_idle_participation(batch)
                self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens
        prefetch_block_table_bound_cpu = draft_input.nxt_kv_lens_cpu

        self._observers.begin_step()

        target_model = self.target_worker.model_runner.model
        sampling_info = batch.sampling_info
        proposal = self._proposal_from_draft_prefetch(
            draft_input=draft_input,
            sampling_info=sampling_info,
        )
        proposal_from_prefetch = proposal is not None
        verify_window = (
            self._verify_window_from_draft_prefetch(draft_input=draft_input, bs=bs)
            if proposal_from_prefetch
            else None
        )
        if verify_window is None:
            verify_window = alloc_verify_window(
                batch=batch,
                bs=bs,
                device=device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
                model_runner=self.model_runner,
            )
        if proposal is None:
            self._wait_for_previous_draft_read()
            with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
                proposal = self._proposer.propose(
                    batch=batch,
                    draft_input=draft_input,
                    verify_window=verify_window,
                    bs=bs,
                    device=device,
                    target_model=target_model,
                    sampling_info=sampling_info,
                )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        confidence = proposal.confidence
        if confidence is None:
            confidence = self._verify_planner.compute_confidence_tensor(
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
                confidence_tap=proposal.confidence_tap,
            )

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and get_parallel().enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        # The upper-bound staging path below is intentionally narrow.  Compact
        # verify and DSV4's C128 interval builder need exact per-row host
        # lengths before ForwardBatch construction, so keep their original
        # synchronization point.  Static K3 verify can enqueue all pointer-
        # stable target metadata from the scheduler's allocation bound first.
        can_stage_target_with_bound = bool(
            seq_lens_cpu_resolver is not None
            and proposal_from_prefetch
            and not run_compact
            and draft_input.nxt_kv_lens_cpu is not None
            and not hasattr(batch.req_to_token_pool, "req_to_c128_sidecar")
        )
        if seq_lens_cpu_resolver is not None:
            if can_stage_target_with_bound:
                self._draft_prefetch_stats["target_prep_deferred"] += 1
            else:
                self._draft_prefetch_stats["target_prep_sync_fallback"] += 1
                seq_lens_cpu_resolver()
                seq_lens_cpu_resolver = None

        verify_ids_2d = (
            draft_input.topk_index
            if proposal_from_prefetch
            else torch.cat([draft_block_ids[:, :1], draft_tokens], dim=1).contiguous()
        )

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        # A live grammar forces the eager path: the folded epilogue accepts inside
        # the cuda graph off its own buffers, where the mask below never lands.
        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            # The epilogue's in-graph accept is greedy (accept_greedy_triton);
            # sampling batches must take the eager accept path even when the
            # draft proposal itself folded.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
            and not batch.has_grammar
        )
        # The non-compact DFlash path rebuilds this immediately before
        # ForwardBatch.init_new().  Compact verify bypasses DFlashVerifyInput,
        # so only that path needs the worker-level hook.
        if run_compact:
            prepare_mamba_track_for_verify(batch)
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    inject_gate=fold_eligible,
                )
            else:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                    seq_lens_cpu_resolver=seq_lens_cpu_resolver,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph
        if batch.has_grammar:
            # run_compact scatters its rows back to (bs * chain_len), so the mask
            # lines up with the logits on both verify paths.
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
            if grammar_mask is not None:
                grammar_mask.apply(logits_output.next_token_logits)

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = fold_eligible and run_compact and can_run_cuda_graph
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        # With prefetch enabled, publish exact post-accept lengths immediately
        # so the private D2H can overlap the remaining forward work.  Feature
        # off deliberately retains the original publish point after logprobs,
        # keeping the baseline execution order unchanged.
        if self.enable_draft_prefetch and on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                accept.out_tokens.reshape(-1),
                chain_stride=self.verify_num_draft_tokens,
            )
        if not self.enable_draft_prefetch and on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        folded_commit = folded_accept and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        next_draft_input = None
        if self.enable_draft_prefetch:
            next_draft_input = make_next_draft_input(
                bonus_tokens=accept.bonus,
                new_seq_lens=accept.new_seq_lens,
            )
            self._schedule_draft_prefetch(
                batch=batch,
                next_draft_input=next_draft_input,
                new_seq_lens=accept.new_seq_lens,
                block_table_bound_cpu=prefetch_block_table_bound_cpu,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        # Keep the critical enqueue path short.  The observer can do CPU
        # bookkeeping (and optional debug D2H staging) after the prefetched
        # Draft ACLGraph/Markov tail is already in flight, so those tasks are
        # covered by useful NPU work instead of delaying its launch.
        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal.folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        if next_draft_input is None:
            next_draft_input = make_next_draft_input(
                bonus_tokens=accept.bonus,
                new_seq_lens=accept.new_seq_lens,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
            expert_distribution_metrics=target_verify.expert_distribution_metrics,
        )

    @diagnostic_stage("commit_kda", device=True)
    def _commit_target_mamba_states_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit the last accepted verify step's KDA/mamba state (chain
        layout: step index = commit_lens - 1) into the persistent caches."""
        if not self._need_mamba_verify_commit:
            return
        # Chain layout only: step index = commit_lens - 1. A tree (topk > 1)
        # layout would need the accept-index mapping the shared spec_utils
        # commit helper does.
        assert get_spec().speculative_eagle_topk in (None, 1)
        attn_backend = self.target_worker.model_runner.attn_backend

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None
        mamba_track_indices = batch.mamba_track_indices

        if mamba_track_indices is not None:
            mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
            seq_lens_cpu = batch.seq_lens_cpu
            if (
                is_npu()
                and seq_lens_cpu is not None
                and seq_lens_cpu.device.type == "cpu"
                and seq_lens_cpu.ndim == 1
                and seq_lens_cpu.numel() == seq_lens_pre_verify.numel()
                and seq_lens_cpu.dtype in (torch.int32, torch.int64)
            ):
                # Verify restores the CPU prefix lengths before the forward.
                # Acceptance can commit at most this many tokens, so this
                # check needs no device readback. Passing None also avoids
                # the NPU backend's conv-state self-copy for untracked rows.
                if all(
                    seq_len >= 0
                    and seq_len // mamba_track_interval
                    == (seq_len + self.verify_num_draft_tokens) // mamba_track_interval
                    for seq_len in seq_lens_cpu.tolist()
                ):
                    mamba_track_indices = None

        if mamba_track_indices is not None:
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
