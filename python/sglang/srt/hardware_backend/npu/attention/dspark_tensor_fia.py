"""Opt-in ACLGraph FIA with device prefix lengths and an external ABI-2 binding.

Prepare each graph shape during warmup; replay passes NPU prefix lengths
unchanged to the native kernel. Build the library and device binary for the
target chip and container ABI. A5 hardware/serving validation is still pending;
see tools/kimi_k3_draft_prefetch.md. No built-in attention operator is replaced.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


class DsparkTensorFIA:
    MAX_BATCH_SIZE = 32

    def __init__(self):
        library = os.environ.get("SGLANG_DSPARK_FIA_TENSOR_LIB", "")
        binary = os.environ.get("SGLANG_DSPARK_FIA_TENSOR_BINARY", "")
        for name, value in (
            ("SGLANG_DSPARK_FIA_TENSOR_LIB", library),
            ("SGLANG_DSPARK_FIA_TENSOR_BINARY", binary),
        ):
            if not value or not Path(value).is_file():
                raise RuntimeError(
                    f"Tensor FIA requires {name} to name an existing build artifact"
                )
        torch.ops.load_library(library)
        self.ops = torch.ops.k3_fia_experimental
        if self.ops.abi_version() != 2:
            raise RuntimeError("Unsupported Tensor FIA binding ABI (expected 2)")
        self.binary = str(Path(binary).resolve())
        self._plans = {}
        # Supported buckets share a conservative workspace bound. Draft layers
        # and replays run in stream order, without a separate planner stream.
        self._workspace = None

    def __call__(self, query, key, value, block_table, prefix_lens, *, scale):
        bs = prefix_lens.numel()
        if not 1 <= bs <= self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Tensor FIA batch must be 1..{self.MAX_BATCH_SIZE}, got {bs}"
            )
        if query.shape != (bs * 7, 4, 64):
            raise ValueError(
                f"Tensor FIA requires Q=[bs*7,4,64], got {tuple(query.shape)}"
            )
        if prefix_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "Tensor FIA binds int32/int64 NPU prefix lengths without a cast"
            )
        key_shape = (
            bs,
            block_table.shape[1],
            prefix_lens.dtype,
            float(scale),
            query.device,
        )
        state = self._plans.get(key_shape)
        if state is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(
                    "Tensor FIA shape was not warmed before ACLGraph capture"
                )
            state_id, workspace_bytes = self.ops.prepare(
                query,
                bs,
                4,
                1,
                128,
                block_table.shape[1],
                prefix_lens.element_size(),
                float(scale),
                self.binary,
            )
            if self._workspace is None:
                self._workspace = torch.empty(
                    workspace_bytes, dtype=torch.uint8, device=query.device
                )
            elif self._workspace.numel() < workspace_bytes:
                raise RuntimeError(
                    "Tensor FIA workspace cannot grow after warming graph inputs"
                )
            state = state_id
            self._plans[key_shape] = state
        output = torch.empty_like(query)
        self.ops.run(
            state, query, key, value, block_table, prefix_lens, output, self._workspace
        )
        return output
