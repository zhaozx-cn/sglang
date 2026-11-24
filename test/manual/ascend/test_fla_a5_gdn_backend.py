"""Manual NPU checks for FLA A5 GDN prefill and Conv1D output layouts.

Run in the SGLang environment after sourcing CANN, with an idle NPU selected:
    ASCEND_RT_VISIBLE_DEVICES=0 python test_fla_a5_gdn_backend.py
"""

import math
import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()

from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu

from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
    AscendGDNAttnBackend,
)
from sglang.srt.model_executor.forward_mode import ForwardMode


def make_backend(lengths, slots=None):
    backend = object.__new__(AscendGDNAttnBackend)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor(cu, dtype=torch.int32, device="npu"),
        mamba_cache_indices=torch.tensor(
            list(range(len(lengths))) if slots is None else slots,
            dtype=torch.int32,
            device="npu",
        ),
    )
    backend.prepare_gdn_inputs(len(lengths), ForwardMode.EXTEND, None)
    return backend


def make_extend_case(
    lengths, *, heads=2, value_dim=128, kernel_size=4, state_len=3, bias=False
):
    slots = list(reversed(range(len(lengths))))
    backend = make_backend(lengths, slots)
    backend.graph_mode = False
    md = backend.forward_metadata
    md.retrieve_next_token = md.retrieve_next_sibling = md.retrieve_parent_token = None
    md.has_mamba_track_mask = False
    value_heads = 2 * heads
    channels = 2 * heads * 128 + value_heads * value_dim
    layer = SimpleNamespace(
        layer_id=0,
        q_dim=heads * 128,
        k_dim=heads * 128,
        v_dim=value_heads * value_dim,
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=value_heads,
        head_q_dim=128,
        head_k_dim=128,
        head_v_dim=value_dim,
        conv_weights=torch.randn(
            channels, kernel_size, device="npu", dtype=torch.bfloat16
        )
        * 0.1,
        bias=(
            torch.randn(channels, device="npu", dtype=torch.bfloat16) * 0.1
            if bias
            else None
        ),
        A_log=torch.zeros(value_heads, device="npu"),
        dt_bias=torch.zeros(value_heads, device="npu", dtype=torch.bfloat16),
    )
    cache = SimpleNamespace(
        conv=[
            torch.randn(
                len(lengths) + 1,
                state_len,
                channels,
                device="npu",
                dtype=torch.bfloat16,
            )
            * 0.1
        ],
        temporal=torch.randn(
            len(lengths) + 1,
            value_heads,
            value_dim,
            128,
            device="npu",
            dtype=torch.float32,
        )
        * 0.01,
    )
    backend.state_layer_cache = lambda layer_id: cache
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens=torch.zeros(len(lengths), device="npu", dtype=torch.int32),
    )
    raw = torch.randn(sum(lengths), channels, device="npu", dtype=torch.bfloat16)
    a = torch.randn(sum(lengths), value_heads, device="npu", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    return backend, layer, cache, batch, raw, a, b


def oracle(q, k, v, g, beta, initial, lengths):
    q = q.cpu().float()
    k = k.cpu().float()
    v = v.cpu().float()
    g = g.cpu().float()
    beta = beta.cpu().float()
    q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).bfloat16().float()
    k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).bfloat16().float()
    groups = v.shape[1] // q.shape[1]
    q = q.repeat_interleave(groups, dim=1) / math.sqrt(q.shape[-1])
    k = k.repeat_interleave(groups, dim=1)
    outputs = torch.empty_like(v)
    final = []
    snapshots = []
    offset = 0
    for seq, length in enumerate(lengths):
        state = initial[seq].cpu().float().clone()
        for token in range(length):
            if token % 64 == 0:
                snapshots.append(state.clone())
            pos = offset + token
            state *= g[pos].exp()[:, None, None]
            delta = beta[pos, :, None] * (v[pos] - (state * k[pos, :, None, :]).sum(-1))
            state += delta[:, :, None] * k[pos, :, None, :]
            outputs[pos] = (state * q[pos, :, None, :]).sum(-1)
        final.append(state)
        offset += length
    return outputs, torch.stack(final), torch.stack(snapshots)


class TestFlaA5GDNBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        torch.set_num_threads(1)
        torch.manual_seed(20261123)

    def assert_numeric(self, name, actual, expected):
        actual = actual.cpu().float()
        expected = expected.cpu().float()
        error = actual - expected
        relative = (
            (error.square().sum() / expected.square().sum().clamp_min(1e-12))
            .sqrt()
            .item()
        )
        maximum = error.abs().max().item()
        print(
            f"METRIC {name}: relative_l2={relative:.6g} max_abs={maximum:.6g}",
            flush=True,
        )
        self.assertTrue(torch.isfinite(actual).all().item(), name)
        self.assertLess(relative, 0.015, name)
        self.assertLess(maximum, 0.05, name)

    def inputs(self, lengths, heads=2, state_dtype=torch.float32):
        total = sum(lengths)
        q = torch.randn(total, heads, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(total, 2 * heads, 128, device="npu", dtype=torch.bfloat16)
        g = -torch.rand(total, 2 * heads, device="npu", dtype=torch.float32) * 0.2
        beta = torch.rand(total, 2 * heads, device="npu", dtype=torch.bfloat16)
        initial = (
            torch.randn(
                len(lengths), 2 * heads, 128, 128, device="npu", dtype=state_dtype
            )
            * 0.1
        )
        return q, k, v, g, beta, initial

    @torch.inference_mode()
    def test_varlen_states_and_layout(self):
        cases = [
            ([1], 2, torch.float32),
            ([63, 64, 65], 2, torch.float32),
            ([127, 128, 129], 2, torch.bfloat16),
            ([65, 3], 4, torch.float32),
            ([65, 3], 16, torch.float32),
            ([0, 65, 0, 3], 2, torch.float32),
        ]
        for lengths, heads, dtype in cases:
            with self.subTest(lengths=lengths, heads=heads, dtype=dtype):
                backend = make_backend(lengths)
                args = self.inputs(lengths, heads, dtype)
                initial_before = args[-1].clone()
                output, final, h = backend.chunk_gdn(*args)
                torch.npu.synchronize()
                expected_output, expected_final, expected_h = oracle(*args, lengths)
                self.assertEqual(tuple(output.shape), (sum(lengths), 2 * heads, 128))
                self.assertEqual(final.dtype, dtype)
                self.assertEqual(
                    tuple(h.shape),
                    (sum((n + 63) // 64 for n in lengths), 2 * heads, 128, 128),
                )
                self.assertEqual(h.dtype, torch.bfloat16)
                self.assertTrue(h.is_contiguous())
                torch.testing.assert_close(args[-1], initial_before, rtol=0, atol=0)
                label = f"{lengths}/{heads}/{dtype}"
                self.assert_numeric(label + "/output", output, expected_output)
                self.assert_numeric(label + "/final", final, expected_final)
                self.assert_numeric(label + "/h", h, expected_h)
                chunk = 0
                for seq, length in enumerate(lengths):
                    if length:
                        # The kernel's FP32->BF16 cast may round ties differently
                        # from torch's cast; allow one BF16 ULP for FP32 seeds.
                        torch.testing.assert_close(
                            h[chunk].float(),
                            initial_before[seq].float(),
                            rtol=0 if dtype == torch.bfloat16 else 0.008,
                            atol=0 if dtype == torch.bfloat16 else 1e-7,
                        )
                        chunk += (length + 63) // 64
                    else:
                        torch.testing.assert_close(
                            final[seq], initial_before[seq], rtol=0, atol=0
                        )

    @torch.inference_mode()
    def test_fused_l2norm_small_and_zero_inputs(self):
        lengths = [65, 3]
        for magnitude in (1e-4, 0.0):
            with self.subTest(magnitude=magnitude):
                q, k, v, g, beta, initial = self.inputs(lengths)
                q = q * magnitude
                k = k * magnitude
                args = (q, k, v, g, beta, initial)
                output, final, h = make_backend(lengths).chunk_gdn(*args)
                expected_output, expected_final, expected_h = oracle(*args, lengths)
                self.assert_numeric(
                    f"small_qk/{magnitude}/output", output, expected_output
                )
                self.assert_numeric(
                    f"small_qk/{magnitude}/final", final, expected_final
                )
                self.assert_numeric(f"small_qk/{magnitude}/h", h, expected_h)

    @torch.inference_mode()
    def test_no_tracking_and_prefix_continuation(self):
        lengths = [129]
        args = self.inputs(lengths)
        output, final, h = make_backend(lengths).chunk_gdn(*args)
        q, k, v, g, beta, initial = args
        suffix = (q[64:], k[64:], v[64:], g[64:], beta[64:], h[1:2].to(initial.dtype))
        resumed, resumed_final, unused_h = make_backend([65]).chunk_gdn(
            *suffix, return_intermediate_states=False
        )
        self.assertIsNone(unused_h)
        self.assert_numeric("prefix_resume/output", resumed, output[64:])
        self.assert_numeric("prefix_resume/final", resumed_final, final)
        _, first_final, _ = make_backend([64]).chunk_gdn(
            q[:64],
            k[:64],
            v[:64],
            g[:64],
            beta[:64],
            initial,
            return_intermediate_states=False,
        )
        continued, continued_final, _ = make_backend([65]).chunk_gdn(
            q[64:],
            k[64:],
            v[64:],
            g[64:],
            beta[64:],
            first_final,
            return_intermediate_states=False,
        )
        self.assert_numeric("chunked_prefill/output", continued, output[64:])
        self.assert_numeric("chunked_prefill/final", continued_final, final)

    @torch.inference_mode()
    def test_forward_extend_state_writeback(self):
        lengths = [65, 3]
        slots = [2, 0]
        backend = make_backend(lengths, slots)
        backend.graph_mode = False
        heads, value_heads, dim = 2, 4, 128
        width = (2 * heads + value_heads) * dim
        raw = torch.randn(sum(lengths), width, dtype=torch.bfloat16, device="npu")
        a = torch.randn(sum(lengths), value_heads, dtype=torch.bfloat16, device="npu")
        b = torch.randn_like(a)
        conv = torch.randn(4, 3, width, dtype=torch.bfloat16, device="npu") * 0.1
        state = (
            torch.randn(4, value_heads, dim, dim, dtype=torch.float32, device="npu")
            * 0.1
        )
        before_state = state.clone()
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=heads * dim,
            k_dim=heads * dim,
            v_dim=value_heads * dim,
            num_q_heads=heads,
            num_k_heads=heads,
            num_v_heads=value_heads,
            head_q_dim=dim,
            head_k_dim=dim,
            head_v_dim=dim,
            conv_weights=torch.randn(width, 4, dtype=torch.bfloat16, device="npu")
            * 0.1,
            bias=None,
            A_log=torch.zeros(value_heads, device="npu"),
            dt_bias=torch.zeros(value_heads, dtype=torch.bfloat16, device="npu"),
        )
        md = backend.forward_metadata
        md.retrieve_next_token = md.retrieve_next_sibling = md.retrieve_parent_token = (
            None
        )
        md.has_mamba_track_mask = True
        md.track_conv_indices = torch.tensor(
            [[61, 62, 63]], dtype=torch.int64, device="npu"
        )
        md.conv_states_mask_indices = torch.tensor([3], dtype=torch.int64, device="npu")
        md.track_ssm_h_src = torch.tensor([1], dtype=torch.int64, device="npu")
        md.track_ssm_h_dst = torch.tensor([3], dtype=torch.int64, device="npu")
        md.track_ssm_recompute_dst = None
        md.track_ssm_final_src = md.track_ssm_final_dst = torch.empty(
            0, dtype=torch.int64, device="npu"
        )
        cache = SimpleNamespace(conv=[conv], temporal=state)
        backend.state_layer_cache = lambda layer_id: cache
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens=torch.tensor([7, 9], dtype=torch.int32, device="npu"),
        )
        convolved = torch.ops.npu.causal_conv1d(
            raw,
            layer.conv_weights.t().contiguous(),
            conv_states=conv.clone(),
            query_start_loc=md.query_start_loc,
            cache_indices=md.mamba_cache_indices,
            has_initial_state=torch.ones(2, dtype=torch.bool, device="npu"),
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )
        q, k, v = convolved.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        g, beta = fused_gdn_gating_npu(layer.A_log, a, b, layer.dt_bias)
        expected, final, h = backend.chunk_gdn(
            q.reshape(-1, heads, dim).contiguous(),
            k.reshape(-1, heads, dim).contiguous(),
            v.reshape(-1, value_heads, dim).contiguous(),
            g,
            beta,
            state[md.mamba_cache_indices],
        )
        actual = backend.forward_extend(layer, batch, raw, a, b)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state[md.mamba_cache_indices], final, rtol=0, atol=0)
        torch.testing.assert_close(state[3], h[1].float(), rtol=0, atol=0)
        torch.testing.assert_close(state[1], before_state[1], rtol=0, atol=0)
        torch.testing.assert_close(conv[3], raw[61:64], rtol=0, atol=0)
        torch.testing.assert_close(conv[2], raw[62:65], rtol=0, atol=0)
        torch.testing.assert_close(conv[0], raw[65:68], rtol=0, atol=0)

    def test_metadata_resets_between_batches(self):
        backend = make_backend([65, 3])
        self.assertEqual(backend.fla_cu_seqlens, [0, 65, 68])
        self.assertEqual(backend.fla_chunk_indices, [0, 0, 0, 1, 1, 0])
        backend.forward_metadata.query_start_loc = torch.tensor(
            [0, 1, 65], dtype=torch.int32, device="npu"
        )
        backend.prepare_gdn_inputs(2, ForwardMode.EXTEND, None)
        self.assertEqual(backend.fla_cu_seqlens, [0, 1, 65])
        self.assertEqual(backend.fla_chunk_indices, [0, 0, 1, 0])
        backend.prepare_gdn_inputs(2, ForwardMode.DECODE, None)
        self.assertIsNone(backend.fla_cu_seqlens)
        self.assertIsNone(backend.fla_chunk_indices)
        self.assertIsNone(backend.fla_empty_seq_indices)

    @torch.inference_mode()
    def test_prefill_head_first_matches_token_first(self):
        cases = [
            ([1, 2, 0, 65], 2, 128, 4, 5, True),
            ([63, 64, 65], 4, 128, 3, 4, False),
            ([65, 3], 16, 128, 2, 2, True),
        ]
        for lengths, heads, value_dim, width, state_len, bias in cases:
            with self.subTest(lengths=lengths, heads=heads, value_dim=value_dim):
                backend, layer, cache, batch, raw, a, b = make_extend_case(
                    lengths,
                    heads=heads,
                    value_dim=value_dim,
                    kernel_size=width,
                    state_len=state_len,
                    bias=bias,
                )
                batch.extend_prefix_lens[::2] = 7
                before_raw = raw.clone()
                expected_conv = cache.conv[0].clone()
                expected_state = cache.temporal.clone()
                conv_tail = expected_conv[:, -(width - 1) :, :].contiguous()
                md = backend.forward_metadata
                convolved = torch.ops.npu.causal_conv1d(
                    raw,
                    backend._get_conv_weights_t(layer),
                    conv_states=conv_tail,
                    bias=layer.bias,
                    query_start_loc=md.query_start_loc,
                    cache_indices=md.mamba_cache_indices,
                    has_initial_state=batch.extend_prefix_lens > 0,
                    activation_mode=1,
                    pad_slot_id=-1,
                    run_mode=0,
                )
                expected_conv[:, -(width - 1) :, :] = conv_tail
                q, k, v = convolved.split(
                    [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
                )
                g, beta = fused_gdn_gating_npu(layer.A_log, a, b, layer.dt_bias)
                expected, final, _ = backend.chunk_gdn(
                    q.reshape(-1, heads, 128).contiguous(),
                    k.reshape(-1, heads, 128).contiguous(),
                    v.reshape(-1, 2 * heads, value_dim).contiguous(),
                    g,
                    beta,
                    expected_state[md.mamba_cache_indices],
                    return_intermediate_states=False,
                )
                expected_state[md.mamba_cache_indices] = final
                actual = backend.forward_extend(layer, batch, raw, a, b)
                torch.npu.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(cache.conv[0], expected_conv, rtol=0, atol=0)
                torch.testing.assert_close(
                    cache.temporal, expected_state, rtol=0, atol=0
                )
                torch.testing.assert_close(raw, before_raw, rtol=0, atol=0)

    @torch.inference_mode()
    def test_prefill_conv_unequal_head_dimensions(self):
        # The installed A5 GDN build only accepts V=128. Check the Conv1D
        # adapter independently for an unequal V width, without invoking GDN.
        backend, layer, cache, batch, raw, a, b = make_extend_case(
            [65, 3], value_dim=256
        )
        md = backend.forward_metadata
        expected_conv = cache.conv[0].clone()
        expected = torch.ops.npu.causal_conv1d(
            raw,
            backend._get_conv_weights_t(layer),
            conv_states=expected_conv,
            query_start_loc=md.query_start_loc,
            cache_indices=md.mamba_cache_indices,
            has_initial_state=batch.extend_prefix_lens > 0,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )
        expected_qkv = expected.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)

        def check_inputs(q, k, v, g, beta, initial, *, qkv_layout, **kwargs):
            self.assertEqual(qkv_layout, "BSND")
            for actual, reference in zip((q, k, v), expected_qkv):
                torch.testing.assert_close(
                    actual.reshape(reference.shape), reference, rtol=0, atol=0
                )
            return v, None, None

        backend.chunk_gdn = check_inputs
        backend.forward_extend(layer, batch, raw, a, b)
        torch.testing.assert_close(cache.conv[0], expected_conv, rtol=0, atol=0)

    @torch.inference_mode()
    def test_prefill_conv_chunked_continuation(self):
        backend, layer, cache, batch, raw, a, b = make_extend_case([129])
        initial_conv = cache.conv[0].clone()
        initial_state = cache.temporal.clone()
        full = backend.forward_extend(layer, batch, raw, a, b)
        full_conv = cache.conv[0].clone()
        full_state = cache.temporal.clone()
        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_state)
        md = backend.forward_metadata
        md.query_start_loc = torch.tensor([0, 64], device="npu", dtype=torch.int32)
        backend.prepare_gdn_inputs(1, ForwardMode.EXTEND, None)
        first = backend.forward_extend(layer, batch, raw[:64], a[:64], b[:64])
        md.query_start_loc = torch.tensor([0, 65], device="npu", dtype=torch.int32)
        backend.prepare_gdn_inputs(1, ForwardMode.EXTEND, None)
        batch.extend_prefix_lens.fill_(64)
        second = backend.forward_extend(layer, batch, raw[64:], a[64:], b[64:])
        self.assert_numeric("conv_chunked/output", torch.cat((first, second)), full)
        self.assert_numeric("conv_chunked/state", cache.temporal, full_state)
        torch.testing.assert_close(cache.conv[0], full_conv, rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
