import logging
from typing import Optional, Union

import torch
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
    move_intermediate_cache,
)

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


class AscendMambaAttnBackendBase(MambaAttnBackendBase):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.state_indices_list_gdn = []

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert max_num_tokens % max_bs == 0, (
            f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        )
        draft_token_num = max_num_tokens // max_bs
        # Static per-bs ReplaySSM decode-ring cursors, captured by pointer and
        # refreshed in place before each graph replay. None when the flag is off.
        self.replayssm_write_pos_list = [] if self._replayssm_enabled() else None
        self.replayssm_force_flush_list = [] if self._replayssm_enabled() else None
        self.replayssm_ring_indices_list = (
            [] if self._replayssm_request_ring() else None
        )
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            if self.replayssm_write_pos_list is not None:
                self.replayssm_write_pos_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            if self.replayssm_force_flush_list is not None:
                self.replayssm_force_flush_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            if self.replayssm_ring_indices_list is not None:
                self.replayssm_ring_indices_list.append(
                    torch.full((i + 1,), -1, dtype=torch.int32, device=self.device)
                )
            self.state_indices_list_gdn.append(
                torch.full(
                    ((i + 1) * draft_token_num,),
                    self.pad_slot_id,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
            ssm_state_indices = torch.arange(
                mamba_indices.shape[0] * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1][
                : len(mamba_indices) * spec_info.draft_token_num
            ].copy_(ssm_state_indices)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        replayssm_write_pos = (
            self.replayssm_write_pos_list[bs - 1]
            if self.replayssm_write_pos_list is not None
            else None
        )
        replayssm_force_flush = (
            self.replayssm_force_flush_list[bs - 1]
            if self.replayssm_force_flush_list is not None
            else None
        )
        replayssm_ring_indices = self._fill_replayssm_ring_indices(bs, req_pool_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
                replayssm_ring_indices=replayssm_ring_indices,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
                replayssm_ring_indices=replayssm_ring_indices,
            )

    def _prepare_replayssm_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        mamba_indices: torch.Tensor,
        forward_mode: ForwardMode,
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: int,
        in_capture: bool,
    ):
        """Mirror the CUDA ring-cursor refresh/advance for an Ascend graph replay.

        The pool owns the live per-slot cursor; the captured kernel reads the
        static per-bs buffer, which is refreshed in place before each replay.
        Advance only on a real decode/idle replay, never during capture or
        target verify.
        """
        ring_indices = self._fill_replayssm_ring_indices(
            bs, req_pool_indices, num_padding
        )
        if self.replayssm_write_pos_list is None:
            return None, None, ring_indices

        mamba_pool = self.state_pool
        write_pos_buf = mamba_pool.replayssm_write_pos
        static_wp = self.replayssm_write_pos_list[bs - 1]
        static_ff = self.replayssm_force_flush_list[bs - 1]
        if write_pos_buf is None:
            return static_wp, static_ff, ring_indices

        slots = mamba_indices.to(torch.long)
        safe_slots = slots.clamp(min=0)
        static_wp[: len(mamba_indices)].copy_(write_pos_buf[safe_slots])

        force_flush_dev = None
        if (
            not mamba_pool.replayssm_is_kda
            and forward_mode.is_decode_or_idle()
            and seq_lens_cpu is not None
        ):
            force_flush_dev = self._replayssm_track_flush_mask(seq_lens_cpu, bs).to(
                device=self.device, dtype=torch.int32
            )
            static_ff.copy_(force_flush_dev)
        else:
            static_ff.zero_()

        if not in_capture and forward_mode.is_decode_or_idle():
            L = mamba_pool.linear_replayssm_cache_len
            valid_mask = slots >= 0
            valid_slots = slots[valid_mask]
            if valid_slots.numel() > 0:
                cur_pos = write_pos_buf[safe_slots]
                flushed = cur_pos == (L - 1)
                if force_flush_dev is not None:
                    flushed = flushed | (force_flush_dev != 0)
                next_pos = torch.where(
                    flushed,
                    torch.zeros_like(cur_pos),
                    (cur_pos + 1) % L,
                )
                uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                next_for_valid = next_pos[valid_mask]
                new_vals = torch.empty(
                    uniq_slots.shape[0],
                    dtype=write_pos_buf.dtype,
                    device=write_pos_buf.device,
                )
                new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                write_pos_buf[uniq_slots] = new_vals

        return static_wp, static_ff, ring_indices

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
        ragged_verify_layout=None,
        *args,
        **kwargs,
    ):
        if ragged_verify_layout is not None:
            raise NotImplementedError(
                "Ascend hybrid linear attention does not support ragged verify"
            )
        # out_graph passes seq_lens_cpu=None at capture; mirror the base guard.
        if seq_lens_cpu is None:
            num_padding = 0
        else:
            num_padding = int(
                torch.count_nonzero(
                    seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
                )
            )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # ReplaySSM pads rows with the kernel's invalid-slot sentinel; the
        # non-replay path keeps the legacy slot-0 padding (masked by
        # query_start_loc downstream). The spec-verify compact ring also relies
        # on the sentinel so the replay kernel skips padded graph rows.
        mamba_pool = self.state_pool
        replayssm_pad = self.replayssm_write_pos_list is not None or (
            mamba_pool is not None
            and getattr(mamba_pool, "replayssm_cache_base", None) is not None
            and not mamba_pool.replayssm_is_kda
        )
        mamba_indices[bs - num_padding :] = self.pad_slot_id if replayssm_pad else 0
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        (
            replayssm_write_pos,
            replayssm_force_flush,
            replayssm_ring_indices,
        ) = self._prepare_replayssm_replay_metadata(
            bs,
            req_pool_indices,
            mamba_indices,
            forward_mode,
            seq_lens_cpu,
            num_padding,
            in_capture,
        )
        track_buf = None
        if mamba_track_indices is not None:
            track_buf = mamba_track_indices
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            ssm_state_indices = torch.arange(
                bs * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1].copy_(ssm_state_indices)
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
                replayssm_ring_indices=replayssm_ring_indices,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
                mamba_track_indices=track_buf,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
                replayssm_ring_indices=replayssm_ring_indices,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0  # Mamba attn does not use seq lens to index kv cache


class AscendMamba2AttnBackend(AscendMambaAttnBackendBase):
    pass


class AscendHybridLinearAttnBackend(HybridLinearAttnBackend):
    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AscendMambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        super().__init__(full_attn_backend, linear_attn_backend, full_attn_layers)

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        state_pool = self.linear_attn_backend.state_pool
        # ReplaySSM spec-verify (GDN compact ring): no intermediate_ssm is
        # allocated; the accepted window lives in the per-request (d, k, g)
        # ring. dspark/dflash call this method directly, while the generic
        # spec_utils commit delegates here too; both advance the same GPU Triton
        # ring cursors and roll back conv with the native Ascend kernel.
        if (
            state_pool is not None
            and state_pool.replayssm_cache_base is not None
            and not state_pool.replayssm_is_kda
        ):
            from sglang.srt.runtime_context import get_spec
            from sglang.srt.speculative.spec_utils import (
                _commit_gdn_replayssm_compact_after_verify,
            )

            if req_pool_indices is None:
                raise ValueError(
                    "Ascend ReplaySSM spec-verify commit requires "
                    "req_pool_indices (the compact ring cursors are keyed by "
                    "request slot)."
                )
            replay_indices = req_pool_indices[:request_number]
            _commit_gdn_replayssm_compact_after_verify(
                mamba_pool=state_pool,
                spec_state=state_pool.get_speculative_mamba2_params_all_layers(),
                state_batch_indices=state_indices_tensor,
                replay_indices=replay_indices,
                accept_lens=last_correct_step_indices + 1,
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                draft_token_num=get_spec().speculative_num_draft_tokens,
            )
            return

        mamba_caches = state_pool.get_speculative_mamba2_params_all_layers()

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        dst_indices_tensor = state_indices_tensor.to(torch.int64)  # [N]
        src_indices_tensor = torch.arange(
            dst_indices_tensor.shape[0],
            device=dst_indices_tensor.device,
            dtype=torch.int64,
        )
        last_steps = last_correct_step_indices.to(torch.int64)  # [N]

        move_intermediate_cache(
            ssm_states,
            intermediate_state_cache,
            dst_indices_tensor,
            src_indices_tensor,
            last_steps,
        )

        draft_token_num = intermediate_state_cache.shape[2]
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            mamba_track_indices = mamba_track_indices.to(torch.int64)
            mamba_steps_to_track = mamba_steps_to_track.to(torch.int64)

            move_intermediate_cache(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                src_indices_tensor,
                mamba_steps_to_track,
            )

            track_mask = mamba_steps_to_track >= 0
            # Track conv state from the verify-time window before rolling back
            # the working slot; NPU does not keep per-step conv intermediates.
            src_slots = torch.where(track_mask, dst_indices_tensor, mamba_track_indices)
            conv_states[:, mamba_track_indices] = conv_states[:, src_slots]

        if dst_indices_tensor.numel() > 0:
            conv_state_rollback(
                conv_states,
                dst_indices_tensor,
                last_steps,
                draft_token_num,
            )

        if mamba_track_indices is not None and mamba_track_indices.numel() > 0:
            conv_state_rollback(
                conv_states,
                mamba_track_indices,
                mamba_steps_to_track,
                draft_token_num,
            )

        return

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass
