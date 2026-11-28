"""Fixed-capacity GDN prefill metadata for breakable NPU graphs.

A captured segment records kernels that read ``cu_seqlens`` / ``chunk_indices``
by tensor address, so those tables must live in capacity-sized buffers that are
allocated once per token bucket and refreshed in place before every replay. This
module mirrors the CUDA contract in
``sglang/srt/layers/attention/linear/gdn_prefill_graph.py`` and the NPU DSV4
precedent in ``dsv4_prefill_graph.py``:

* ``max_seqs = min(num_tokens, cap)`` sequences,
* ``max_chunks = cdiv(num_tokens, 64) + max_seqs`` delta-rule chunks (every
  sequence contributes one partial chunk on top of its full ones),
* fixed ``max_seqs`` tracking tables for the prefix-cache snapshots.

``fill`` rebuilds the sequence/chunk tables from the host-side extend lengths the
scheduler already owns (small host-to-device copies, no device-to-host read, no
synchronisation). Real rows come first. The unused tail is padded with empty
sequences, and padded chunk rows point just past sequence 0 (``valid <= 0``), so
a chunk kernel that masks zero-length chunks returns immediately.

Tracking (``track=True``, only when the model runs with the extra mamba buffer)
gets fixed-length tables as well: rows that track nothing point at the reserved
state-pool row 0, so a replay always copies ``max_seqs`` rows and only padding is
written for the rows this batch does not use.

.. note::

   The filled tables are only safe when the installed FLA operator clamps its task
   space to the chunks ``cu_seqlens`` describes (padded rows must never be
   decoded). ``fla_npu.ops.ascendc.GDN_PADDED_METADATA_SUPPORTED`` is that
   capability bit; ``fla_gdn_padded_metadata_supported()`` is the single gate
   callers use, and the arch35 kernels set it.
"""

from __future__ import annotations

from typing import Optional

import torch

# Delta-rule chunk size the GDN extend is launched with.
GDN_CHUNK_SIZE = 64

# The mamba state pool allocates ``size + 1`` rows and never hands out row 0 (the
# slot allocators start at 1), so capacity rows that must not touch live state
# point at it. The eager path uses the same placeholder for padded requests.
RESERVED_PADDING_SLOT = 0


def chunk_capacity(num_tokens: int, max_seqs: int) -> int:
    """Worst-case chunk count for ``max_seqs`` sequences totalling ``num_tokens``."""
    return (int(num_tokens) + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE + max(1, int(max_seqs))


def fla_gdn_padded_metadata_supported() -> bool:
    """Whether the installed FLA GDN operator can consume padded capacity tables."""

    try:
        from fla_npu.ops import ascendc
    except Exception:  # pragma: no cover - FLA absent on non-Ascend hosts
        return False
    return bool(getattr(ascendc, "GDN_PADDED_METADATA_SUPPORTED", False))


def canonical_chunk_indices(extend_seq_lens) -> list[int]:
    """Flattened ``(seq_id, local_chunk_id)`` pairs in sequence-major order."""

    pairs: list[int] = []
    for seq, length in enumerate(int(x) for x in extend_seq_lens):
        for chunk in range((length + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE):
            pairs.extend((seq, chunk))
    return pairs


class GDNPrefillGraphMetadata:
    """Stable, capacity-sized GDN tables read by one captured prefill bucket."""

    def __init__(self) -> None:
        self.num_tokens = 0
        self.max_seqs = 0
        self.max_chunks = 0
        self.device: Optional[torch.device] = None
        # Tables the captured GDN body reads by address.
        self.cu_seqlens: Optional[torch.Tensor] = None  # int64 [max_seqs + 1]
        self.chunk_indices: Optional[torch.Tensor] = None  # int64 [2 * max_chunks]
        self.num_chunks: Optional[torch.Tensor] = None  # int64 [1], real Nc
        # Capacity-row helpers for the eager (non-captured) parts of the layer.
        self.empty_mask: Optional[torch.Tensor] = None  # bool [max_seqs]
        self.cache_indices: Optional[torch.Tensor] = None  # [max_seqs]
        self.has_initial_state: Optional[torch.Tensor] = None  # bool [max_seqs]
        # Prefix-cache tracking, one row per capacity sequence.
        self.track = False
        self.conv_len = 0
        self.track_conv_indices: Optional[torch.Tensor] = None  # int64 [max_seqs, conv_len]
        self.track_conv_dst: Optional[torch.Tensor] = None  # int64 [max_seqs]
        self.track_h_row: Optional[torch.Tensor] = None  # int64 [max_seqs]
        self.track_h_dst: Optional[torch.Tensor] = None  # int64 [max_seqs]
        self.track_final_src: Optional[torch.Tensor] = None  # int64 [max_seqs]
        self.track_final_dst: Optional[torch.Tensor] = None  # int64 [max_seqs]

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        max_seqs_cap: int,
        device: torch.device,
        cache_index_dtype: torch.dtype,
        conv_len: int = 0,
        track: bool = False,
    ) -> "GDNPrefillGraphMetadata":
        state = cls()
        state.num_tokens = int(num_tokens)
        state.max_seqs = max(1, min(state.num_tokens, int(max_seqs_cap)))
        state.max_chunks = chunk_capacity(state.num_tokens, state.max_seqs)
        state.device = device
        state.cu_seqlens = torch.zeros(
            state.max_seqs + 1, dtype=torch.int64, device=device
        )
        state.chunk_indices = torch.zeros(
            2 * state.max_chunks, dtype=torch.int64, device=device
        )
        state.num_chunks = torch.zeros(1, dtype=torch.int64, device=device)
        # Everything starts empty/padded; fill() rewrites the real rows.
        state.empty_mask = torch.ones(state.max_seqs, dtype=torch.bool, device=device)
        state.cache_indices = torch.full(
            (state.max_seqs,),
            RESERVED_PADDING_SLOT,
            dtype=cache_index_dtype,
            device=device,
        )
        state.has_initial_state = torch.zeros(
            state.max_seqs, dtype=torch.bool, device=device
        )

        state.track = bool(track)
        state.conv_len = max(1, int(conv_len))
        if state.track:
            rows = state.max_seqs
            int64 = {"dtype": torch.int64, "device": device}
            state.track_conv_indices = torch.zeros((rows, state.conv_len), **int64)
            state.track_conv_dst = torch.full(
                (rows,), RESERVED_PADDING_SLOT, **int64
            )
            state.track_h_row = torch.zeros((rows,), **int64)
            state.track_h_dst = torch.full((rows,), RESERVED_PADDING_SLOT, **int64)
            state.track_final_src = torch.zeros((rows,), **int64)
            state.track_final_dst = torch.full(
                (rows,), RESERVED_PADDING_SLOT, **int64
            )
        return state

    def can_fit(self, extend_seq_lens) -> bool:
        lens = [int(x) for x in extend_seq_lens]
        return len(lens) <= self.max_seqs and sum(lens) <= self.num_tokens

    def fill(
        self,
        *,
        extend_seq_lens,
        cache_indices: Optional[torch.Tensor] = None,
        extend_prefix_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Rebuild every table for this batch (called before each replay).

        ``cache_indices`` / ``extend_prefix_lens`` are the live batch's device
        tensors; only their first ``bs`` rows are read.
        """

        lens = [int(x) for x in extend_seq_lens]
        bs = len(lens)
        if not self.can_fit(lens):
            raise ValueError(
                f"GDN prefill graph batch of {bs} sequence(s) / {sum(lens)} tokens "
                f"exceeds the captured bucket ({self.max_seqs} / {self.num_tokens})"
            )
        total = sum(lens)

        # cu_seqlens: real boundaries, then empty sequences at the tail.
        cu = [0] * (self.max_seqs + 1)
        running = 0
        for index, length in enumerate(lens):
            running += length
            cu[index + 1] = running
        for index in range(bs + 1, self.max_seqs + 1):
            cu[index] = total

        # chunk_indices: canonical rows first, then rows just past sequence 0.
        pairs = canonical_chunk_indices(lens)
        n_chunks = len(pairs) // 2
        pad_chunk = (lens[0] + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE if bs else 0
        chunk_rows = pairs + [0, pad_chunk] * (self.max_chunks - n_chunks)

        self.cu_seqlens.copy_(torch.tensor(cu, dtype=torch.int64))
        self.chunk_indices.copy_(torch.tensor(chunk_rows, dtype=torch.int64))
        self.num_chunks.fill_(n_chunks)

        # Zero-length sequences (and the padded tail) keep their incoming state.
        empty_mask = [(index >= bs) or lens[index] == 0 for index in range(self.max_seqs)]
        self.empty_mask.copy_(torch.tensor(empty_mask, dtype=torch.bool))

        if cache_indices is not None:
            self.cache_indices[:bs].copy_(
                cache_indices[:bs].to(dtype=self.cache_indices.dtype)
            )
            self.cache_indices[bs:].fill_(RESERVED_PADDING_SLOT)
        if extend_prefix_lens is not None:
            torch.gt(extend_prefix_lens[:bs], 0, out=self.has_initial_state[:bs])
            self.has_initial_state[bs:].fill_(False)

    def fill_track(self, plan: Optional[dict]) -> None:
        """Refresh the fixed-length tracking tables for this batch.

        ``plan`` carries the per-batch index tensors the eager metadata builds
        (``track_conv_indices`` / ``conv_states_mask_indices`` / ``track_ssm_*``),
        or None when this batch tracks nothing. Rows past the plan keep no-op
        values (source 0, destination = the reserved padding row), so a captured
        replay always copies the same number of rows and touches only padding for
        the rows this batch does not use.
        """

        if not self.track:
            return
        plan = plan or {}

        def copy_rows(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
            dst.fill_(0)
            if src is None or src.numel() == 0:
                return
            rows = min(int(src.shape[0]), int(dst.shape[0]))
            if rows > 0:
                dst[:rows].copy_(src[:rows].to(dtype=dst.dtype))

        def copy_dst(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
            dst.fill_(RESERVED_PADDING_SLOT)
            if src is None or src.numel() == 0:
                return
            rows = min(int(src.shape[0]), int(dst.shape[0]))
            if rows > 0:
                dst[:rows].copy_(src[:rows].to(dtype=dst.dtype))

        copy_rows(self.track_conv_indices, plan.get("conv_indices"))
        copy_dst(self.track_conv_dst, plan.get("conv_dst"))
        copy_rows(self.track_h_row, plan.get("h_row"))
        copy_dst(self.track_h_dst, plan.get("h_dst"))
        copy_rows(self.track_final_src, plan.get("final_src"))
        copy_dst(self.track_final_dst, plan.get("final_dst"))

    def apply_track(self, ssm_states: torch.Tensor, h: Optional[torch.Tensor]) -> None:
        """Copy the tracked intermediate/final states (captured replay path).

        Same two copies the eager ``_track_mamba_state_extend`` performs, but over
        the fixed-length tables, so the captured segment has no per-batch shapes.
        """

        if not self.track or h is None:
            return
        ssm_states[self.track_h_dst] = h[self.track_h_row].to(
            ssm_states.dtype, copy=False
        )
        ssm_states[self.track_final_dst] = ssm_states[self.track_final_src]
