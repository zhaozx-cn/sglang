"""DFLASH spec-v2 overlap scheduling data structures."""

import contextlib
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dspark_components.dspark_diagnostics import (
    diagnostic_stage,
    get_diagnostics,
    tensor_description,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils.common import is_pin_memory_available

_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """Return an optional plan stream/context for overlap scheduling prep kernels."""
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


@dataclass
class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations (spec-v2)."""

    # Legacy Eagle-shaped fields; DFLASH relays via FutureMap so these are unused.
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    nxt_kv_lens_cpu: Optional[torch.Tensor] = None
    nxt_kv_lens_sum: Optional[int] = None
    _prepare_batch_seq_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_gpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_gpu_buf: Optional[torch.Tensor] = None

    # DSPARK draft-prefetch validity lives on CPU deliberately. Checking a
    # validity bit embedded in topk_p with Tensor.item() serializes the NPU
    # forward stream once per decode step. The scheduler only reads this tiny
    # host mirror while filtering/merging batches; the token payload itself is
    # still relayed through FutureMap on device.
    draft_prefetch_valid_cpu: Optional[torch.Tensor] = None
    # Exact post-verify lengths, when a legacy prefetch path already copied
    # them. The device-seq DSPark path deliberately leaves this unset.
    draft_prefetch_seq_lens_cpu: Optional[torch.Tensor] = None
    # CPU allocation upper bound prepared by the scheduler one iteration
    # earlier. It only sizes the draft block table; attention uses the exact
    # device prefix lengths and never treats this value as logical seq_lens.
    draft_prefetch_block_table_bound_cpu: Optional[torch.Tensor] = None
    # Verify-window metadata prepared together with the prefetched proposal.
    # Reusing it on the next round avoids a second cache_loc_update for the
    # same request rows and sequence positions.
    draft_prefetch_positions_2d: Optional[torch.Tensor] = None
    draft_prefetch_verify_cache_loc_2d: Optional[torch.Tensor] = None
    draft_prefetch_cancelled: bool = False
    # The completed proposal stays on this next_draft_input.  FutureMap still
    # relays the bonus token and sequence length, but skips the redundant
    # topk scatter/gather for this direct handoff.
    draft_prefetch_direct: bool = False

    # Filled by scheduler after dispatch.
    future_indices: Optional[torch.Tensor] = None

    verify_token_budget: Optional[int] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)
        # Spec v2 draft state itself does not change token accounting.
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

    def wait_draft_prefetch(self) -> None:
        """Compatibility no-op: device-seq prefetch is enqueued on the main stream."""

    def release_draft_prefetch_plan(self, *, cancel: bool = False) -> None:
        if cancel:
            self.draft_prefetch_cancelled = True

    def cancel_draft_prefetch(self) -> None:
        """Cancel a proposal whose row layout is about to change."""
        self.release_draft_prefetch_plan(cancel=True)

    def discard_draft_prefetch(self) -> None:
        """Cancel a proposal for a batch that is being dropped entirely."""
        self.release_draft_prefetch_plan(cancel=True)

    def _ensure_prepare_length_buffers(
        self, bs: int, device: torch.device | str
    ) -> None:
        pin_memory = is_pin_memory_available(device)

        def needs_cpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

        # The three CPU scratch buffers grow together; capacity is the only
        # invariant (batch is int64 non-pinned, cur/nxt are int32 pinned).
        if needs_cpu_alloc(self._prepare_batch_seq_lens_cpu_buf):
            capacity = grown_capacity(self._prepare_batch_seq_lens_cpu_buf)
            self._prepare_batch_seq_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int64, device="cpu"
            )
            self._prepare_cur_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_nxt_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )

        if needs_gpu_alloc(self._prepare_cur_kv_lens_gpu_buf):
            capacity = grown_capacity(self._prepare_cur_kv_lens_gpu_buf)
            self._prepare_cur_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )
            self._prepare_nxt_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
        )

    @diagnostic_stage("kv_prepare", device=True)
    def prepare_for_decode(self, batch: ScheduleBatch):
        """Allocate headroom in the shared req_to_token pool for the next DFLASH step.

        DFLASH spec-v2 uses overlap scheduling's "over-allocation" approach: we reserve
        future KV slots ahead of time so the worker can gather `out_cache_loc` directly
        from `req_to_token` without allocator backup/restore. CPU metadata intentionally
        lags by one iteration; keep it separate from the reserved upper bound that backs
        the overallocated mapping.
        """
        bs = batch.batch_size()
        if bs == 0:
            self.cancel_draft_prefetch()
            return

        batch.maybe_evict_swa()

        self._ensure_prepare_length_buffers(bs, batch.device)
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_gpu_buf is not None
        assert self._prepare_nxt_kv_lens_gpu_buf is not None
        batch_seq_lens_cpu_t = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu_t = self._prepare_cur_kv_lens_cpu_buf[:bs]
        nxt_kv_lens_cpu_t = self._prepare_nxt_kv_lens_cpu_buf[:bs]

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = int(get_spec().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )
        # Match ScheduleBatch's memory admission estimate. DSPark prefetch
        # needs the next window before the next allocation, in addition to
        # the two windows required by ordinary overlap scheduling.
        reserve = get_alloc_reserve_per_decode()
        page_size = batch.token_to_kv_pool_allocator.page_size

        cur_kv_lens_host, nxt_kv_lens_host, num_needed_tokens = (
            page_aligned_decode_alloc_lens(
                batch.reqs,
                reserve=reserve,
                page_size=page_size,
            )
        )

        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        nxt_kv_lens_sum = 0
        committed_seq_lens_sum = 0
        for i, (req, cur, nxt) in enumerate(
            zip(batch.reqs, cur_kv_lens_host, nxt_kv_lens_host)
        ):
            committed_len = int(req.kv.kv_committed_len)
            committed_seq_lens_sum += committed_len
            top_k = int(req.sampling_params.top_k)

            batch_seq_lens_cpu_t[i] = committed_len
            cur_kv_lens_cpu_t[i] = cur
            nxt_kv_lens_cpu_t[i] = nxt

            nxt_kv_lens_sum += nxt
            if top_k > max_top_k:
                max_top_k = top_k
            if i == 0:
                uniform_top_k_value = top_k
            elif uniform_top_k and top_k != uniform_top_k_value:
                uniform_top_k = False

        self.max_top_k = max(max_top_k, 1)
        self.uniform_top_k_value = uniform_top_k_value if uniform_top_k else None

        # Keep the complete planning operation ordered on the caller stream.
        # Length tensors are consumed only when at least one request grows its
        # reserved KV range.  Skip both H2D copies on the common zero-growth
        # page step for every batch size.  On a growth step retain the original
        # pinned-copy path; unlike extra fill kernels it has the same ordering
        # and kernel surface as the proven allocator path.
        #
        # Retain the zero-growth call because it updates request-side allocation
        # bookkeeping even when no new block is needed.
        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)
        caller_stream = None
        if plan_stream is not None:
            caller_stream = torch.get_device_module(batch.device).current_stream()

        cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
        nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
        diag = get_diagnostics()
        if diag is not None:
            diag.emit(
                "kv_allocation_plan",
                needed_tokens=num_needed_tokens,
                reserve=reserve,
                page_size=page_size,
                cur_host=cur_kv_lens_host,
                next_host=nxt_kv_lens_host,
                cur_device=tensor_description(cur_kv_lens),
                next_device=tensor_description(nxt_kv_lens),
                separate_plan_stream=plan_stream is not None,
            )

        if plan_stream is not None:
            with plan_stream_ctx:
                plan_stream.wait_stream(caller_stream)
                if num_needed_tokens > 0:
                    token = diag.begin("kv_length_h2d") if diag is not None else None
                    cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
                    nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)
                    if diag is not None:
                        diag.checkpoint("kv_length_h2d")
                        diag.end(token)
                alloc_for_spec_decode(
                    batch.tree_cache,
                    batch.req_to_token_pool,
                    reqs=batch.reqs,
                    req_pool_indices=batch.req_pool_indices,
                    cur_kv_lens=cur_kv_lens,
                    cur_kv_lens_cpu=cur_kv_lens_cpu_t,
                    nxt_kv_lens=nxt_kv_lens,
                    nxt_kv_lens_cpu=nxt_kv_lens_cpu_t,
                    num_needed_tokens=num_needed_tokens,
                    batch=batch,
                )
            # Forward work cannot observe partially prepared req_to_token/KV
            # allocation state from the opt-in full plan stream.
            caller_stream.wait_stream(plan_stream)
        else:
            if num_needed_tokens > 0:
                token = diag.begin("kv_length_h2d") if diag is not None else None
                cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
                nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)
                if diag is not None:
                    diag.checkpoint("kv_length_h2d")
                    diag.end(token)

            # Allocation and shared req_to_token writes deliberately remain on
            # the caller stream to preserve cross-rank DeepEP launch order.
            alloc_for_spec_decode(
                batch.tree_cache,
                batch.req_to_token_pool,
                reqs=batch.reqs,
                req_pool_indices=batch.req_pool_indices,
                cur_kv_lens=cur_kv_lens,
                cur_kv_lens_cpu=cur_kv_lens_cpu_t,
                nxt_kv_lens=nxt_kv_lens,
                nxt_kv_lens_cpu=nxt_kv_lens_cpu_t,
                num_needed_tokens=num_needed_tokens,
                batch=batch,
            )
        for req in batch.reqs:
            req.decode_batch_idx += 1
        # Seed committed; overlap's resolve overwrites it with the published value.
        batch.seq_lens_cpu = batch_seq_lens_cpu_t
        batch.seq_lens_sum = committed_seq_lens_sum
        self.nxt_kv_lens_cpu = nxt_kv_lens_cpu_t
        self.nxt_kv_lens_sum = nxt_kv_lens_sum
        # Device-seq draft prefetch was already enqueued on the forward stream;
        # no host Future or plan-ready handoff is required here.

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ):
        # A row reorder invalidates the process-local proposal/window.  Cancel
        # it before touching the payload and let the worker use the normal
        # proposer for the changed batch.
        self.cancel_draft_prefetch()
        self.draft_prefetch_block_table_bound_cpu = None
        if self.draft_prefetch_valid_cpu is not None:
            valid_indices = (
                new_indices_cpu
                if new_indices_cpu is not None
                else new_indices.to(device="cpu").tolist()
            )
            self.draft_prefetch_valid_cpu = self.draft_prefetch_valid_cpu[valid_indices]
            # The queued proposal belongs to the old row layout. Force the
            # changed batch through the normal proposer.
            self.draft_prefetch_valid_cpu.zero_()
            if self.draft_prefetch_seq_lens_cpu is not None:
                self.draft_prefetch_seq_lens_cpu = self.draft_prefetch_seq_lens_cpu[
                    valid_indices
                ]
            if (
                self.future_indices is None
                and self.draft_prefetch_positions_2d is not None
            ):
                self.draft_prefetch_positions_2d = self.draft_prefetch_positions_2d[
                    new_indices
                ]
            if (
                self.future_indices is None
                and self.draft_prefetch_verify_cache_loc_2d is not None
            ):
                self.draft_prefetch_verify_cache_loc_2d = (
                    self.draft_prefetch_verify_cache_loc_2d[new_indices]
                )
        if self.nxt_kv_lens_cpu is not None:
            if new_indices_cpu is not None:
                self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices_cpu]
            else:
                self.nxt_kv_lens_cpu = self.nxt_kv_lens_cpu[new_indices.cpu()]
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())

        if self.future_indices is not None:
            # The window belongs to the exact prior batch row layout. A
            # filter/reorder invalidates that direct handoff; the worker will
            # rebuild it once for the changed batch.
            self.draft_prefetch_positions_2d = None
            self.draft_prefetch_verify_cache_loc_2d = None
            if self.draft_prefetch_direct:
                self.topk_p = self.topk_p[new_indices]
                self.topk_index = self.topk_index[new_indices]
            self.future_indices = self.future_indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        # A mixed batch has a different row layout.  Discard both process-local
        # proposals and fall back to the normal proposer for this iteration.
        self.cancel_draft_prefetch()
        spec_info.cancel_draft_prefetch()
        self.draft_prefetch_block_table_bound_cpu = None
        if (
            self.draft_prefetch_valid_cpu is not None
            or spec_info.draft_prefetch_valid_cpu is not None
        ):
            left_valid = self.draft_prefetch_valid_cpu
            if left_valid is None:
                left_valid = torch.zeros(self.topk_p.shape[0], dtype=torch.bool)
            right_valid = spec_info.draft_prefetch_valid_cpu
            if right_valid is None:
                right_valid = torch.zeros(spec_info.topk_p.shape[0], dtype=torch.bool)
            self.draft_prefetch_valid_cpu = torch.cat(
                [
                    left_valid,
                    right_valid,
                ]
            )
            self.draft_prefetch_valid_cpu.zero_()
        if (
            self.draft_prefetch_seq_lens_cpu is not None
            and spec_info.draft_prefetch_seq_lens_cpu is not None
        ):
            self.draft_prefetch_seq_lens_cpu = torch.cat(
                [
                    self.draft_prefetch_seq_lens_cpu,
                    spec_info.draft_prefetch_seq_lens_cpu,
                ]
            )
        else:
            # A mixed batch containing a fresh prefill row has no complete
            # prefetched host mirror. Let FutureMap resolve the whole batch via
            # its normal path.
            self.draft_prefetch_seq_lens_cpu = None

        if self.future_indices is None and (
            self.draft_prefetch_positions_2d is not None
            and spec_info.draft_prefetch_positions_2d is not None
            and self.draft_prefetch_verify_cache_loc_2d is not None
            and spec_info.draft_prefetch_verify_cache_loc_2d is not None
        ):
            self.draft_prefetch_positions_2d = torch.cat(
                [
                    self.draft_prefetch_positions_2d,
                    spec_info.draft_prefetch_positions_2d,
                ]
            )
            self.draft_prefetch_verify_cache_loc_2d = torch.cat(
                [
                    self.draft_prefetch_verify_cache_loc_2d,
                    spec_info.draft_prefetch_verify_cache_loc_2d,
                ]
            )
        else:
            self.draft_prefetch_positions_2d = None
            self.draft_prefetch_verify_cache_loc_2d = None

        if self.nxt_kv_lens_cpu is not None:
            assert spec_info.nxt_kv_lens_cpu is not None
            self.nxt_kv_lens_cpu = torch.cat(
                [self.nxt_kv_lens_cpu, spec_info.nxt_kv_lens_cpu]
            )
            self.nxt_kv_lens_sum = int(self.nxt_kv_lens_cpu.sum().item())
        elif spec_info.nxt_kv_lens_cpu is not None:
            self.nxt_kv_lens_cpu = spec_info.nxt_kv_lens_cpu
            self.nxt_kv_lens_sum = spec_info.nxt_kv_lens_sum

        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            if self.draft_prefetch_direct or spec_info.draft_prefetch_direct:
                self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
                self.topk_index = torch.cat(
                    [self.topk_index, spec_info.topk_index], dim=0
                )
                self.draft_prefetch_direct = True
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
