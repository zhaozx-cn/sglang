"""Qwen3.5 causal decode FIA: paged KV, ragged batches and graph updates.

Run on Ascend A5 with pytest after selecting an available NPU. No model weights
or distributed process group are needed; both entry points are real backend calls.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    ForwardMetadata,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or not is_npu_arch35(), reason="requires Ascend A5"
)


def make_case(lengths, heads=(16, 2), extra_queries=0):
    generator = torch.Generator().manual_seed(20261124)
    batch_size, page_size, dim = len(lengths), 128, 256
    q_heads, kv_heads = heads
    blocks_per_request = (max(lengths) + page_size - 1) // page_size
    blocks = batch_size * blocks_per_request
    q_cpu = torch.randn(
        batch_size + extra_queries, q_heads, dim, generator=generator
    ).to(torch.bfloat16)
    k_cpu = torch.randn(blocks, page_size, kv_heads, dim, generator=generator).to(
        torch.bfloat16
    )
    v_cpu = torch.randn(blocks, page_size, kv_heads, dim, generator=generator).to(
        torch.bfloat16
    )
    # Non-contiguous physical pages catch accidental dense-cache assumptions.
    pages = torch.randperm(blocks, generator=generator).view(batch_size, -1)
    table_cpu = torch.zeros(batch_size, 2048, dtype=torch.int32)
    table_cpu[:, :blocks_per_request] = pages.to(torch.int32)
    q = q_cpu.flatten(1).npu()
    k = k_cpu.flatten(0, 1).npu()
    v = v_cpu.flatten(0, 1).npu()
    backend = AscendAttnBackend.__new__(AscendAttnBackend)
    backend.use_mla = False
    backend.use_fia = True
    backend.use_fa = False
    backend.use_alibi = False
    backend.use_quant_flash_attn = False
    backend.is_hybrid_swa = False
    backend.graph_mode = False
    backend.enable_torch_compile = False
    backend.dense_mla_dcp_target = False
    backend.page_size = page_size
    backend.fia_mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1).npu()
    backend.forward_metadata = ForwardMetadata(
        block_tables=table_cpu.npu(), seq_lens_cpu_list=list(lengths)
    )
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: k, get_value_buffer=lambda _: v
    )
    layer = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        tp_v_head_num=kv_heads,
        qk_head_dim=dim,
        v_head_dim=dim,
        scaling=dim**-0.5,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
        logit_cap=0,
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    def reference(lens):
        result = torch.zeros_like(q_cpu, dtype=torch.float32)
        for row, length in enumerate(lens):
            if length == 0:
                continue
            ids = table_cpu[row, : (length + page_size - 1) // page_size].long()
            keys = k_cpu[ids].flatten(0, 1)[:length].float()
            values = v_cpu[ids].flatten(0, 1)[:length].float()
            keys = keys.repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1)
            values = values.repeat_interleave(q_heads // kv_heads, dim=1).transpose(
                0, 1
            )
            scores = q_cpu[row].float().unsqueeze(1) @ keys.transpose(1, 2)
            result[row] = ((scores * layer.scaling).softmax(-1) @ values).squeeze(1)
        return result.flatten(1)

    return backend, q, layer, batch, reference


def check_output(output, reference, lengths):
    output = output.float().cpu()
    assert torch.isfinite(output).all()
    active = torch.tensor([i for i, length in enumerate(lengths) if length > 0])
    # Allow BF16 rounding against FP32; also compare to the old FIA path below.
    torch.testing.assert_close(output[active], reference[active], rtol=0.03, atol=0.004)


@pytest.mark.parametrize("lengths", [[1], [128], [129, 17], [11275, 513, 0, 0]])
@pytest.mark.parametrize("heads", [(16, 2), (8, 1)])
def test_eager_paged_decode(lengths, heads):
    backend, q, layer, batch, reference = make_case(lengths, heads)
    expected = reference(lengths)
    outputs = []
    for use_fast_path in (False, True):
        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_backend.is_npu_arch35",
            return_value=use_fast_path,
        ):
            output = backend.forward_decode(
                q, None, None, layer, batch, save_kv_cache=False
            )
        check_output(output, expected, lengths)
        outputs.append(output.cpu())
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0.01, atol=0.0001)


def test_eager_restores_query_padding():
    lengths = [129, 17]
    backend, q, layer, batch, reference = make_case(lengths, extra_queries=2)
    output = backend.forward_decode(q, None, None, layer, batch, save_kv_cache=False)
    assert output.shape == q.shape
    check_output(output, reference(lengths), lengths)
    assert torch.count_nonzero(output[2:]).item() == 0
    with patch(
        "sglang.srt.hardware_backend.npu.attention.ascend_backend.is_npu_arch35",
        return_value=False,
    ):
        baseline = backend.forward_decode(
            q, None, None, layer, batch, save_kv_cache=False
        )
    torch.testing.assert_close(output, baseline, rtol=0.01, atol=0.0001)


@pytest.mark.parametrize("heads", [(16, 2), (8, 1)])
def test_graph_replay_refreshes_kv_lengths(heads):
    backend, q, layer, batch, reference = make_case([11300] * 4, heads)
    backend.forward_metadata.seq_lens_cpu_list = [1] * 4
    for _ in range(2):
        backend.forward_decode(q, None, None, layer, batch, save_kv_cache=False)
    torch.npu.synchronize()

    backend.graph_mode = True
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = backend.forward_decode(
            q, None, None, layer, batch, save_kv_cache=False
        )

    with ThreadPoolExecutor(
        max_workers=1,
        initializer=torch.npu.set_device,
        initargs=(torch.npu.current_device(),),
    ) as pool:
        for lengths in ([11275, 513, 0, 0], [129, 17, 2, 0], [1, 11300, 128, 1]):
            update = pool.submit(
                graph.update, cpu_update_input=[{"actual_seq_lengths_kv": lengths}]
            )
            graph.replay()
            update.result(timeout=30)
            torch.npu.synchronize()
            check_output(output, reference(lengths), lengths)
            captured_output = output.cpu()
            backend.graph_mode = False
            backend.forward_metadata.seq_lens_cpu_list = list(lengths)
            eager = backend.forward_decode(
                q, None, None, layer, batch, save_kv_cache=False
            ).cpu()
            backend.graph_mode = True
            torch.testing.assert_close(captured_output, eager, rtol=0, atol=0)
