"""Fine-grained Ascend dual-stream scheduling for MoE shared experts.

Ascend NPUs execute matrix products on the Cube unit and elementwise /
reduction / quantization work on the Vector (AIV) unit. The dense shared
expert and the routed MoE both contain Cube and Vector stages, so simply
putting the whole shared expert on a second stream makes the two branches
contend for the same unit. The scheme ported from vllm-ascend instead splits
the shared expert into its stages and aligns each auxiliary-stream stage with a
main-stream stage that uses the *other* unit:

    main stream:  router(gate, Cube) -> topk/routing(Vector) -> routed(Cube)
    aux  stream:  input prep(Vector) -> shared Gate-Up(Cube) -> act(Vector)
                  -> shared Down(Cube)

Concretely:

* Shared input preparation (clone/cast/quant metadata) runs on the auxiliary
  stream while the Cube-heavy router matmul runs on the main stream.
* Shared Gate-Up waits for ``router_output_ready`` so its Cube work does not
  compete with the router Cube, then covers the main-stream TopK/routing and
  dispatch window.
* Shared activation is Vector work and is queued right after Gate-Up, so it
  overlaps the routed Cube matmuls; the shared Down projection then overlaps
  the routed Vector tail.

Everything is expressed with streams and events only: there is no host-side
synchronization and no data-dependent Python branch, so the same code path is
captured by ACL graph replay and executed eagerly when graphs are disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

# One auxiliary stream per device, shared by every MoE layer: layers run
# sequentially, so a per-layer stream would only waste HCCL/ACL resources.
_DUAL_STREAMS: dict[int, "torch.npu.Stream"] = {}


def _get_dual_stream() -> "torch.npu.Stream":
    device = torch.npu.current_device()
    stream = _DUAL_STREAMS.get(device)
    if stream is None:
        stream = torch.npu.Stream()
        _DUAL_STREAMS[device] = stream
    return stream


@dataclass(frozen=True)
class PreparedSharedInput:
    """Shared-expert activation whose Vector preparation is already queued.

    ``ready_event`` describes scheduling only; it is the point on the
    auxiliary stream after which the activation may be consumed.
    """

    hidden_states: torch.Tensor
    ready_event: Optional["torch.npu.Event"]


@dataclass(frozen=True)
class SubmittedSharedGateUp:
    """Shared Gate-Up already queued on the auxiliary stream."""

    gate_up: torch.Tensor
    done_event: Optional["torch.npu.Event"]


class FineGrainedMoEDualStream:
    """Schedule a shared-expert MLP around the routed MoE on Ascend.

    The executor owns no weights: callers pass the shared MLP stages as
    callables. This keeps the scheduling policy independent of the model's
    MLP class and quantization method, and makes it shareable by any model
    that has a dense shared expert next to a routed MoE layer.

    ``enabled=False`` keeps the exact same call sequence on the main stream,
    so callers do not need a separate non-overlapped path.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self._stream = _get_dual_stream()
            # Reused across forwards: each event is consumed before the next
            # forward records it again (the join at the end orders the tail).
            self._input_ready = torch.npu.Event()
            self._router_ready = torch.npu.Event()
            self._gate_up_ready = torch.npu.Event()
            self._shared_ready = torch.npu.Event()
        else:
            self._stream = None
            self._input_ready = None
            self._router_ready = None
            self._gate_up_ready = None
            self._shared_ready = None

    @property
    def stream(self) -> Optional["torch.npu.Stream"]:
        return self._stream

    def maybe_record_event(self) -> Optional["torch.npu.Event"]:
        """Record a main-stream milestone for the auxiliary stream.

        Returns ``None`` when overlap is disabled so consumers can skip the
        wait and pay no host-side event cost.
        """
        if not self.enabled:
            return None
        return torch.npu.current_stream().record_event()

    def prepare_input(
        self,
        hidden_states: torch.Tensor,
        prepare: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> PreparedSharedInput:
        """Run Vector-only shared input preparation on the auxiliary stream.

        The auxiliary stream is forked from the current stream, so the
        preparation overlaps the router matmul that the caller runs next on
        the main stream. ``prepare`` must not enqueue Cube work.
        """
        if not self.enabled:
            prepared = prepare(hidden_states) if prepare is not None else hidden_states
            return PreparedSharedInput(prepared, None)

        aux = self._stream
        aux.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(aux):
            prepared = prepare(hidden_states) if prepare is not None else hidden_states
            self._input_ready.record()
        return PreparedSharedInput(prepared, self._input_ready)

    def record_router_output(self) -> Optional["torch.npu.Event"]:
        """Record the point where the main-stream router matmul is queued.

        Shared Gate-Up waits on it so its Cube work starts after the router
        Cube work instead of competing with it.
        """
        if not self.enabled:
            return None
        self._router_ready.record()
        return self._router_ready

    def submit_gate_up(
        self,
        prepared: PreparedSharedInput,
        gate_up_fn: Callable[[torch.Tensor], torch.Tensor],
        after: Optional["torch.npu.Event"] = None,
    ) -> SubmittedSharedGateUp:
        """Queue the shared Gate-Up matmul on the auxiliary stream."""
        if not self.enabled:
            return SubmittedSharedGateUp(gate_up_fn(prepared.hidden_states), None)

        aux = self._stream
        with torch.npu.stream(aux):
            if prepared.ready_event is not None:
                aux.wait_event(prepared.ready_event)
            if after is not None:
                aux.wait_event(after)
            gate_up = gate_up_fn(prepared.hidden_states)
            self._gate_up_ready.record()
        return SubmittedSharedGateUp(gate_up, self._gate_up_ready)

    def finish(
        self,
        submitted: SubmittedSharedGateUp,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        down_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        hidden_states: Optional[torch.Tensor] = None,
        apply_gate_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        after: Optional["torch.npu.Event"] = None,
    ) -> tuple[torch.Tensor, Optional["torch.npu.Event"]]:
        """Queue the shared activation + Down projection on the aux stream.

        The Vector activation overlaps the routed Cube matmuls still running
        on the main stream; the optional dense shared-expert gate is applied
        on the auxiliary stream so it does not add a main-stream Cube op
        before the join.
        """
        if not self.enabled:
            out = down_fn(act_fn(submitted.gate_up))
            if apply_gate_fn is not None:
                out = apply_gate_fn(hidden_states, out)
            return out, None

        aux = self._stream
        with torch.npu.stream(aux):
            if submitted.done_event is not None:
                aux.wait_event(submitted.done_event)
            if after is not None:
                aux.wait_event(after)
            out = down_fn(act_fn(submitted.gate_up))
            if apply_gate_fn is not None:
                out = apply_gate_fn(hidden_states, out)
            self._shared_ready.record()
        return out, self._shared_ready

    def join(self, ready_event: Optional["torch.npu.Event"] = None) -> None:
        """Join the auxiliary stream before the caller consumes its output."""
        if not self.enabled:
            return
        current = torch.npu.current_stream()
        if ready_event is not None:
            current.wait_event(ready_event)
        else:
            current.wait_stream(self._stream)
