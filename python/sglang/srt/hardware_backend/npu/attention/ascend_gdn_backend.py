from typing import Optional, Tuple, Union

import torch
from sgl_kernel_npu.operators.attention.gdn import (
    fused_gdn_gating,
    fused_gdn_gating_without_sigmoid,
    recurrent_gated_delta_rule,
)

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput


def _canonical_chunk_indices(cu_seqlens, chunk_size: int):
    """Flattened (seq_id, local_chunk_id) pairs in canonical sequence-major order.

    This is the layout the FLA chunk kernels expect for ``chunk_indices``: one
    pair per chunk, sequences in order and chunks ascending inside a sequence.
    Empty sequences contribute no pairs.
    """

    indices = []
    for seq, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk in range((end - begin + chunk_size - 1) // chunk_size):
            indices.extend((seq, chunk))
    return indices


class AscendGDNAttnBackend(AscendMambaAttnBackendBase):
    """Gated-delta-net linear attention on Ascend NPU.

    Breakable prefill graphs (``--cuda-graph-backend-prefill=breakable``) keep
    the GDN extend out of the captured segments. The FLA chunk launcher takes
    ``cu_seqlens`` / ``chunk_indices`` as int64 **device** tensors (the kernel
    reads them from GM and host tiling only uses their shapes), but the number
    of chunks is data dependent and sizes the FLA workspace, so the extent of
    the metadata cannot vary inside a captured segment. ``supports_prefill_graph_extend``
    therefore stays False and ``RadixLinearAttention`` calls back into this
    backend through its ``@eager_on_graph`` break: the surrounding segments
    replay from graph-pool buffers while the GDN body re-runs eagerly against
    the live batch metadata that ``init_forward_metadata`` rebuilt for this
    replay.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = torch.Size(
            (
                *model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[
                    :-2
                ],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-1],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-2],
            )
        )
        backends = model_runner.linear_attn_backends
        decode_backend = backends.decode
        prefill_backend = backends.prefill
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def supports_prefill_graph_extend(self) -> bool:
        # The FLA chunk count is data dependent and sizes its workspace, so the
        # metadata extent would freeze to the capture bucket's sequence layout.
        # Keep the per-layer eager break.
        return False

    def _prepare_mamba_track_metadata(self, forward_batch: ForwardBatch):
        if self.forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[mamba_track_mask_indices]
            )

    @staticmethod
    def _host_cu_seqlens(forward_batch, query_start_loc):
        """Host mirror of the varlen table, used only to size ``chunk_indices``.

        ``query_start_loc`` lives on the device and the canonical chunk list has
        a data-dependent length, so its extent cannot be derived on device
        without a device-to-host read (and that length sizes the FLA tiling).
        SGLang already owns the extend lengths on the host, and ``extend_start_loc``
        is exactly their running sum, so the two representations only need to be
        reconciled when the layout is known to differ (two-batch overlap) or the
        CPU lengths are unavailable.
        """

        lens = (
            getattr(forward_batch, "extend_seq_lens_cpu", None)
            if forward_batch is not None
            else None
        )
        tbo_range = (
            getattr(forward_batch, "tbo_parent_token_range", None)
            if forward_batch is not None
            else None
        )
        expected = int(query_start_loc.numel()) - 1
        if lens is not None and len(lens) == expected and tbo_range is None:
            cu = [0]
            for length in lens:
                cu.append(cu[-1] + int(length))
            return cu
        return query_start_loc.to("cpu", dtype=torch.int64).tolist()

    def prepare_gdn_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accept_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            # indices
            self.ssm_state_indices = torch.arange(
                cache_indices.shape[0] * seq_len,
                dtype=torch.int32,
                device=cache_indices.device,
            )
        else:
            self.ssm_state_indices = cache_indices
        if forward_mode.is_prefill():
            self.actual_seq_lengths = torch.diff(
                self.forward_metadata.query_start_loc
            ).to(torch.int32)

        # FLA consumes cu_seqlens/chunk_indices as int64 *device* tensors now: the
        # kernel reads them from GM and host tiling only uses their shapes. Build
        # them once per forward, never per GDN layer.
        #   * cu_seqlens mirrors query_start_loc on the device (no host int array
        #     and no device-to-host synchronisation).
        #   * chunk_indices keeps a data-dependent length (it sizes the FLA
        #     workspace), so it is materialised from the host-side extend lengths
        #     the scheduler already owns and uploaded once per forward.
        # Decode and target verify keep their existing paths.
        # Breakable prefill graphs never reach this at capture time: metadata
        # init runs outside the segments and the GDN body itself replays as an
        # eager break, so these tables always describe the live batch.
        self.fla_cu_seqlens = None
        self.fla_chunk_indices = None
        self.fla_empty_seq_indices = None
        if forward_mode.is_prefill() and not forward_mode.is_target_verify():
            query_start_loc = self.forward_metadata.query_start_loc
            host_cu_seqlens = self._host_cu_seqlens(forward_batch, query_start_loc)
            chunk_indices = _canonical_chunk_indices(host_cu_seqlens, 64)
            if chunk_indices:
                self.fla_cu_seqlens = query_start_loc.to(dtype=torch.int64)
                self.fla_chunk_indices = torch.tensor(
                    chunk_indices,
                    dtype=torch.int64,
                    device=query_start_loc.device,
                )
            # An all-empty batch has no chunk rows. Keep both tables absent so the
            # launcher follows its historical degenerate path instead of handing
            # the operator an empty chunk_indices tensor (which it rejects).
            empty_sequences = [
                seq
                for seq, (begin, end) in enumerate(
                    zip(host_cu_seqlens, host_cu_seqlens[1:])
                )
                if begin == end
            ]
            if empty_sequences:
                self.fla_empty_seq_indices = torch.tensor(
                    empty_sequences, dtype=torch.int64, device=cache_indices.device
                )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata(forward_batch)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = False

    def _get_conv_weights_t(self, layer: RadixLinearAttention) -> torch.Tensor:
        w = getattr(layer, "_conv_weights_t", None)
        if w is None:
            w = layer.conv_weights.transpose(0, 1).contiguous()
            layer._conv_weights_t = w
        return w

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = torch.ops.npu.causal_conv1d(
            mixed_qkv,
            self._get_conv_weights_t(layer),
            conv_states=conv_states,
            bias=layer.bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.state_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0
        if is_target_verify:
            num_token_padding = mixed_qkv.shape[0]
            if (
                not self.graph_mode
                and forward_batch.global_num_token_non_padded_cpu != num_token_padding
            ):
                mixed_qkv = mixed_qkv[: forward_batch.global_num_token_non_padded_cpu]
                a = a[: forward_batch.global_num_token_non_padded_cpu]
                b = b[: forward_batch.global_num_token_non_padded_cpu]
                seq_len = forward_batch.global_num_token_non_padded_cpu

            batch_size = cache_indices.shape[0]
            draft_token_num = forward_batch.spec_info.draft_token_num
            num_accepted_tokens = torch.full(
                (batch_size,),
                draft_token_num,
                dtype=torch.int32,
                device=mixed_qkv.device,
            )
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accepted_tokens,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )
        else:
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[forward_metadata.track_conv_indices]
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )
            kernel_size = layer.conv_weights.shape[-1]
            conv_states_for_prefill = conv_states[
                :, -(kernel_size - 1) :, :
            ].contiguous()

            # A single head-first output can pack Q/K/V when their head
            # dimensions match. Keep token-first output for unequal widths.
            conv_head_num = (
                layer.num_q_heads + layer.num_k_heads + layer.num_v_heads
                if layer.head_q_dim == layer.head_k_dim == layer.head_v_dim
                else 0
            )
            qkv_layout = "BNSD" if conv_head_num else "BSND"
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                bias=layer.bias,
                conv_states=conv_states_for_prefill,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_states,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
                output_layout="HEAD_FIRST" if conv_head_num else "TOKEN_FIRST",
                head_num=conv_head_num,
            )
            conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        if is_target_verify:
            g, beta = fused_gdn_gating_without_sigmoid(layer.A_log, a, b, layer.dt_bias)
            beta = beta.unsqueeze(0)
            num_heads, head_k_dim = layer.num_q_heads, layer.head_q_dim
            num_value_heads, head_v_dim = layer.num_v_heads, layer.head_v_dim

            mixed_qkv_last_dim = mixed_qkv.shape[-1]

            mixed_qkv = mixed_qkv.view(batch_size, -1, mixed_qkv_last_dim)
            beta = beta.view(batch_size, -1, num_value_heads)
            g = g.view(batch_size, -1, num_value_heads)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update(
                mixed_qkv,
                num_heads,
                num_value_heads,
                head_k_dim,
                head_v_dim,
                recurrent_state=ssm_states,
                beta=beta,
                g=g,
                cache_indices=cache_indices,
                intermediate_state=intermediate_state_cache,
            )
            core_attn_out = core_attn_out.view(-1, num_value_heads, head_v_dim)
            if (not self.graph_mode) and core_attn_out.shape[0] < num_token_padding:
                core_attn_out = torch.cat(
                    [
                        core_attn_out,
                        core_attn_out.new_zeros(
                            num_token_padding - core_attn_out.shape[0],
                            *core_attn_out.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        else:
            if qkv_layout == "BNSD":
                query, key, value = mixed_qkv.split(
                    [layer.num_q_heads, layer.num_k_heads, layer.num_v_heads], dim=0
                )
            else:
                query, key, value = torch.split(
                    mixed_qkv,
                    [layer.q_dim, layer.k_dim, layer.v_dim],
                    dim=-1,
                )

                actual_seq_len = query.shape[0]
                query = query.view(
                    actual_seq_len, layer.num_q_heads, layer.head_q_dim
                ).contiguous()
                key = key.view(
                    actual_seq_len, layer.num_k_heads, layer.head_k_dim
                ).contiguous()
                value = value.view(
                    actual_seq_len, layer.num_v_heads, layer.head_v_dim
                ).contiguous()
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

            core_attn_out, last_recurrent_state, h = self.chunk_gdn(
                query,
                key,
                value,
                g,
                beta,
                ssm_states[cache_indices],
                return_intermediate_states=forward_metadata.has_mamba_track_mask,
                qkv_layout=qkv_layout,
            )

            if last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out

    def chunk_gdn(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        *,
        return_intermediate_states=True,
        qkv_layout="BSND",
    ):
        from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd

        # The FLA prepare kernel normalizes raw Q/K. Do not normalize them
        # again here. use_exp2 selects internal arithmetic; g remains the
        # natural-log decay produced by fused_gdn_gating.
        g = g.to(torch.float32)  # [T, Nv]

        # Variable-length requests share physical B=1. SGL Conv1D already
        # writes head-first Q/K/V; g and beta always remain token-first.
        # cu_seqlens/chunk_indices are int64 device tensors prepared once per
        # forward in prepare_gdn_inputs (the FLA kernel reads them from GM).
        outputs = npu_chunk_gated_delta_rule_fwd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=self.fla_cu_seqlens,
            chunk_indices=self.fla_chunk_indices,
            scale=k.shape[-1] ** -0.5,
            use_exp2=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            # FLA 26.10 keeps fused QK normalization but omits backward-only
            # g_cumsum/A/QK auxiliary outputs when recomputation is allowed.
            # Prefix snapshots are controlled independently below.
            disable_recompute=False,
            return_intermediate_states=return_intermediate_states,
            state_v_first=True,
            layout=qkv_layout,
        )
        core_attn_out, last_recurrent_state = outputs[:2]
        if self.fla_empty_seq_indices is not None:
            # FLA does not write final-state rows for zero-length sequences.
            # Such requests must retain their incoming state in the pool.
            last_recurrent_state.index_copy_(
                0,
                self.fla_empty_seq_indices,
                initial_state.index_select(0, self.fla_empty_seq_indices),
            )
        h = outputs[5]
        if h is not None:
            # FLA 26.10: [1, NT, Hv, V, K]; prefix tracking: [NT, Hv, V, K].
            # FLA stores snapshots in the activation dtype; the existing
            # tracking code converts selected snapshots to the state dtype.
            h = h.squeeze(0).contiguous()
        return core_attn_out.squeeze(0), last_recurrent_state, h

    def fused_recurrent_gated_delta_rule_update(
        self,
        mix_qkv: torch.Tensor,
        num_heads,
        num_value_heads,
        head_k_dim,
        head_v_dim,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state: Optional[torch.Tensor] = None,
    ):
        beta = beta.to(torch.bfloat16)
        g = g.to(torch.float32)
        batch_size = mix_qkv.shape[0]
        seq_len = mix_qkv.shape[1]
        scale = 1 / (head_k_dim**0.5)

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )

        if self.graph_mode:
            num_accept_tokens = torch.full(
                [batch_size], 1, dtype=torch.int32, device=cache_indices.device
            )
            actual_seq_lengths = torch.full(
                [batch_size], seq_len, dtype=torch.int32, device=cache_indices.device
            )
            ssm_state_indices = self.forward_metadata.mamba_cache_indices_gdn
        else:
            num_accept_tokens = self.num_accept_tokens
            actual_seq_lengths = self.actual_seq_lengths
            ssm_state_indices = self.ssm_state_indices

        attn_core_out = recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accept_tokens,
            g=g,
        )

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, seq_len, num_value_heads, head_k_dim, head_v_dim
            )
        return attn_core_out
