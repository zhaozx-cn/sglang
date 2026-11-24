"""Fine-grained Ascend dual-stream scheduling for MoE shared experts.

Ascend NPUs execute matrix products on the Cube unit and elementwise /
reduction / quantization work on the Vector (AIV) unit. The dense shared
expert and the routed MoE both contain Cube and Vector stages, so simply
putting the whole shared expert on a second stream makes the two branches
contend for the same unit. This module ports the fine-grained scheme used by
vllm-ascend: the shared expert is split into its stages and each auxiliary
stream stage is aligned with a main-stream stage that uses the *other* unit.

For a Qwen2-MoE style layer the timeline is::

    main stream:  router(Cube) -> topk(Vector) -> GMM1(Cube) -> act(Vector)
                  -> GMM2(Cube) -> combine(Vector)
    aux  stream:      quant(Vector)   ->   Gate-Up(Cube)  -> act(Vector)
                  -> Down(Cube)

with these ordering constraints:

* ``prepare_input`` forks the auxiliary stream before the router matmul and
  runs only Vector work (activation quantization), overlapping the Cube-heavy
  router.
* Shared Gate-Up waits for ``router_output_ready`` so its Cube work covers the
  routed TopK/dispatch window instead of competing with the router Cube.
* Shared activation waits for ``routed_gmm2_start``; it is Vector work, so it
  overlaps the routed GMM2 Cube matmul without competing with routed GMM1's
  Vector activation/quantization.
* Shared Down (plus its optional input preparation) waits for
  ``routed_combine_start``, overlapping the routed combine.

The routed milestones are published by :class:`RoutedMoEMilestones` through a
``ContextVar``: the model enters :meth:`FineGrainedMoEDualStream.routed_milestones`
around the routed experts call, and the Ascend MoE runner records the stage
boundaries with :func:`record_routed_gmm2_start` /
:func:`record_routed_combine_start`. Routed implementations that run as one
fused operator (and therefore expose no intermediate stages) simply record
nothing; the shared expert then runs as a single concurrent auxiliary-stream
workload, matching the reference implementation.

Everything is expressed with streams and events only: there is no host-side
synchronization and no data-dependent Python branch, so the same code path is
captured by ACL graph replay and executed eagerly when graphs are disabled.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

import torch

# A prepared activation is either a plain tensor or a pre-quantized
# ``(quantized, scale)`` pair that the NPU linear method consumes directly.
MoELinearInput = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

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


@dataclass
class RoutedMoEMilestones:
    """Routed-path stage boundaries consumed by the shared-expert stream.

    The events are preallocated by :class:`FineGrainedMoEDualStream` and only
    *recorded* at run time, so replaying a captured graph allocates nothing.
    The ``*_recorded`` flags distinguish "the routed backend exposed this
    boundary" from "the event object merely exists".
    """

    shared_input_ready: Optional["torch.npu.Event"] = None
    router_output_ready: Optional["torch.npu.Event"] = None
    routed_gmm2_start: Optional["torch.npu.Event"] = None
    routed_combine_start: Optional["torch.npu.Event"] = None
    gmm2_recorded: bool = False
    combine_recorded: bool = False

    @property
    def has_fine_grained_stage_events(self) -> bool:
        """Whether the routed implementation exposed internal boundaries.

        A partially populated set still counts as fine-grained so consumers
        fail fast on the missing boundary instead of hiding a producer bug.
        """
        return self.gmm2_recorded or self.combine_recorded


# The milestones of the routed MoE call currently being enqueued. ContextVar
# keeps the publication thread- and nesting-safe; it never crosses into the
# graph itself (the recorder calls happen in Python while the graph is built).
_CURRENT_MILESTONES: ContextVar[Optional[RoutedMoEMilestones]] = ContextVar(
    "sglang_npu_moe_dual_stream_milestones", default=None
)


def record_routed_gmm2_start() -> None:
    """Record the routed GMM2 start for a published milestones context.

    Called by the Ascend MoE runner right before the down (GMM2) grouped
    matmul. It is a no-op when the dual stream is disabled, so the runner pays
    only a single attribute check on the normal path.
    """
    milestones = _CURRENT_MILESTONES.get()
    if milestones is None or milestones.routed_gmm2_start is None:
        return
    milestones.routed_gmm2_start.record()
    milestones.gmm2_recorded = True


def record_routed_combine_start() -> None:
    """Record the routed token-combine start for a published milestones context.

    Called by the Ascend post-permute hooks immediately before the routed
    output is combined back into ``[tokens, hidden]``.
    """
    milestones = _CURRENT_MILESTONES.get()
    if milestones is None or milestones.routed_combine_start is None:
        return
    milestones.routed_combine_start.record()
    milestones.combine_recorded = True


@dataclass(frozen=True)
class PreparedSharedInput:
    """Shared-expert input whose Vector preparation is already queued.

    ``hidden_states`` stays the raw activation used by the output gate, while
    ``gate_up_input`` is what the Gate-Up linear consumes (a pre-quantized
    tuple when the input scheme supports overlapping quantization).
    """

    hidden_states: torch.Tensor
    gate_up_input: MoELinearInput
    ready_event: Optional["torch.npu.Event"] = None


@dataclass(frozen=True)
class SubmittedSharedGateUp:
    """Shared Gate-Up already queued on the auxiliary stream."""

    gate_up: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True)
class SubmittedSharedActivation:
    """Shared activation (and its input quantization) queued on the aux stream."""

    activated: torch.Tensor
    hidden_states: torch.Tensor


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
            self._gmm2_ready = torch.npu.Event()
            self._combine_ready = torch.npu.Event()
            self._shared_ready = torch.npu.Event()
        else:
            self._stream = None
            self._input_ready = None
            self._router_ready = None
            self._gmm2_ready = None
            self._combine_ready = None
            self._shared_ready = None

    @property
    def stream(self) -> Optional["torch.npu.Stream"]:
        return self._stream

    def new_milestones(self) -> Optional[RoutedMoEMilestones]:
        """Create the milestone bundle for one routed MoE call."""
        if not self.enabled:
            return None
        return RoutedMoEMilestones(
            shared_input_ready=self._input_ready,
            router_output_ready=self._router_ready,
            routed_gmm2_start=self._gmm2_ready,
            routed_combine_start=self._combine_ready,
        )

    @contextmanager
    def routed_milestones(
        self, milestones: Optional[RoutedMoEMilestones]
    ) -> Iterator[None]:
        """Publish the milestones for the routed experts call inside the block."""
        if not self.enabled or milestones is None:
            yield
            return
        token = _CURRENT_MILESTONES.set(milestones)
        try:
            yield
        finally:
            _CURRENT_MILESTONES.reset(token)

    def prepare_input(
        self,
        hidden_states: torch.Tensor,
        prepare: Optional[Callable[[torch.Tensor], MoELinearInput]] = None,
        milestones: Optional[RoutedMoEMilestones] = None,
    ) -> PreparedSharedInput:
        """Run Vector-only shared input preparation on the auxiliary stream.

        The auxiliary stream is forked from the current stream, so the
        preparation overlaps the router matmul that the caller runs next on
        the main stream. ``prepare`` must not enqueue Cube work.
        """
        if not self.enabled:
            gate_up_input = (
                prepare(hidden_states) if prepare is not None else hidden_states
            )
            return PreparedSharedInput(hidden_states, gate_up_input)

        aux = self._stream
        aux.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(aux):
            gate_up_input = (
                prepare(hidden_states) if prepare is not None else hidden_states
            )
            self._input_ready.record()
        if milestones is not None:
            milestones.shared_input_ready = self._input_ready
        return PreparedSharedInput(hidden_states, gate_up_input, self._input_ready)

    def record_router_output(
        self, milestones: Optional[RoutedMoEMilestones] = None
    ) -> Optional["torch.npu.Event"]:
        """Record the point where the main-stream router matmul is queued.

        Shared Gate-Up waits on it so its Cube work starts after the router
        Cube work instead of competing with it.
        """
        if not self.enabled:
            return None
        self._router_ready.record()
        if milestones is not None:
            milestones.router_output_ready = self._router_ready
        return self._router_ready

    def submit_gate_up(
        self,
        prepared: PreparedSharedInput,
        gate_up_fn: Callable[[MoELinearInput], torch.Tensor],
        milestones: Optional[RoutedMoEMilestones] = None,
    ) -> SubmittedSharedGateUp:
        """Queue the shared Gate-Up matmul on the auxiliary stream."""
        if not self.enabled:
            return SubmittedSharedGateUp(
                gate_up_fn(prepared.gate_up_input), prepared.hidden_states
            )

        after = milestones.router_output_ready if milestones is not None else None
        with torch.npu.stream(self._stream):
            if after is not None:
                self._stream.wait_event(after)
            gate_up = gate_up_fn(prepared.gate_up_input)
        return SubmittedSharedGateUp(gate_up, prepared.hidden_states)

    @staticmethod
    def _wait_for_stage(
        milestones: Optional[RoutedMoEMilestones],
        event: Optional["torch.npu.Event"],
        recorded: bool,
        name: str,
    ) -> None:
        """Wait for a required routed milestone on the current (aux) stream.

        Fused routed implementations expose no internal boundaries; the shared
        MLP then stays independent and runs as one concurrent workload, which
        is exactly what the reference implementation does.
        """
        if milestones is None or not milestones.has_fine_grained_stage_events:
            return
        if not recorded:
            raise RuntimeError(
                f"Missing {name} event while the Ascend fine-grained MoE "
                "dual stream is enabled."
            )
        torch.npu.current_stream().wait_event(event)

    def submit_activation(
        self,
        submitted: SubmittedSharedGateUp,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        milestones: Optional[RoutedMoEMilestones] = None,
    ) -> SubmittedSharedActivation:
        """Queue the shared activation after the routed GMM2 has started.

        The activation is Vector work, so running it alongside the routed GMM2
        Cube matmul overlaps the two units without competing with routed GMM1's
        own Vector activation/quantization.
        """
        if not self.enabled:
            return SubmittedSharedActivation(
                act_fn(submitted.gate_up), submitted.hidden_states
            )

        with torch.npu.stream(self._stream):
            self._wait_for_stage(
                milestones,
                milestones.routed_gmm2_start if milestones is not None else None,
                milestones.gmm2_recorded if milestones is not None else False,
                "routed_gmm2_start",
            )
            activated = act_fn(submitted.gate_up)
        return SubmittedSharedActivation(activated, submitted.hidden_states)

    def submit_down(
        self,
        submitted: SubmittedSharedActivation,
        prepare_down_fn: Callable[[torch.Tensor], MoELinearInput],
        down_fn: Callable[[MoELinearInput], torch.Tensor],
        milestones: Optional[RoutedMoEMilestones] = None,
        apply_gate_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> tuple[torch.Tensor, Optional["torch.npu.Event"]]:
        """Queue the shared Down projection after the routed combine starts.

        The Down input preparation (typically the activation quantization) is
        still Vector work and runs while routed GMM2 is in flight; only the
        Cube matmul itself waits for ``routed_combine_start``.
        """
        if not self.enabled:
            down_input = prepare_down_fn(submitted.activated)
            out = down_fn(down_input)
            if apply_gate_fn is not None:
                out = apply_gate_fn(submitted.hidden_states, out)
            return out, None

        with torch.npu.stream(self._stream):
            down_input = prepare_down_fn(submitted.activated)
            self._wait_for_stage(
                milestones,
                milestones.routed_combine_start if milestones is not None else None,
                milestones.combine_recorded if milestones is not None else False,
                "routed_combine_start",
            )
            out = down_fn(down_input)
            if apply_gate_fn is not None:
                out = apply_gate_fn(submitted.hidden_states, out)
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
