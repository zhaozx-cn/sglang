"""Opt-in DSPark submission/completion diagnostics; never a performance mode.

Disabled decorators return the original function at import time. No diagnostic
stream, collective, D2H, or tensor-value inspection is added. Unique events on
the existing streams identify completed prefixes. A CPU observer uses AscendCL
directly so querying progress does not drain torch_npu's dispatch queue.
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

_DIRECTORY = os.getenv("SGLANG_DSPARK_DIAG_DIR", "")
_diag = None


def get_diagnostics():
    return _diag if _diag is not None and _diag.active else None


def configure_diagnostics(device, *, tp_rank, dp_rank):
    global _diag
    if _DIRECTORY:
        _diag = DsparkDiagnostics(device, tp_rank=tp_rank, dp_rank=dp_rank)


def start_diagnostics():
    if _diag is not None:
        _diag.start()


def tensor_description(tensor):
    if tensor is None:
        return None
    # Metadata only: in particular, never .cpu(), .item(), or repr(tensor).
    return {
        "ptr": tensor.data_ptr(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _batch_description(args, kwargs):
    batch = kwargs.get("forward_batch")
    if batch is None:
        batch = kwargs.get("batch")
    if batch is None and len(args) > 1 and hasattr(args[1], "forward_mode"):
        batch = args[1]
    if batch is None:
        return {}
    mode = getattr(batch, "forward_mode", None)
    reqs = getattr(batch, "reqs", None)
    counts = getattr(batch, "original_global_num_tokens_cpu", None)
    if counts is None:
        counts = getattr(batch, "global_num_tokens", None)
    return {
        "mode": getattr(mode, "name", str(mode)),
        "bs": len(reqs) if reqs is not None else getattr(batch, "batch_size", None),
        "global_counts": counts if isinstance(counts, (list, tuple)) else None,
        "forward_iter": getattr(batch, "forward_iter", None),
    }


def diagnostic_stage(name, *, device=False, iteration=False, graph=False):
    """Record host entry/exit and, optionally, an existing-stream checkpoint."""

    def decorate(fn):
        if not _DIRECTORY:
            return fn

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            diag = get_diagnostics()
            if diag is None:
                return fn(*args, **kwargs)
            details = _batch_description(args, kwargs)
            stage = name
            if iteration:
                diag.step += 1
            if graph:
                role = "draft" if args[0].model_runner.is_draft_worker else "target"
                stage = role + "_graph"
            token = diag.begin(stage, **details)
            try:
                if graph:
                    diag.graph_reuse(args[0], role)
                result = fn(*args, **kwargs)
                event = diag.checkpoint(stage) if device else None
                if graph:
                    diag.graph_submitted(args[0], role, event)
                diag.end(token)
                return result
            except BaseException as exc:
                diag.emit(
                    "exception",
                    stage=stage,
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                )
                diag.end(token, failed=True)
                raise

        return wrapped

    return decorate


class DsparkDiagnostics:
    def __init__(self, device, *, tp_rank, dp_rank):
        self.device = device
        self.active = False
        self.step = 0
        self.rank = {
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        phases = os.getenv("SGLANG_DSPARK_DIAG_SYNC_PHASES", "")
        self.sync_phases = {p.strip() for p in phases.split(",") if p.strip()}
        allowed = {
            "target_graph",
            "draft_graph",
            "target_reuse",
            "draft_reuse",
            "accept",
            "commit_kda",
            "commit_hidden",
            "draft_propose",
            "draft_prefetch",
            "worker_forward",
            "seq_lens_d2h",
        }
        if self.sync_phases - allowed:
            raise ValueError(
                f"Unknown DSPark diagnostic sync phases: {self.sync_phases - allowed}"
            )
        self.lock = threading.RLock()
        self.serial = 0
        self.open_spans = {}
        self.events = []
        self.graph_events = {}
        self.last_complete = {}
        self.event_limit = 8192
        self.main_thread = threading.get_ident()
        directory = Path(_DIRECTORY)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank{tp_rank:03d}_pid{os.getpid()}.jsonl"
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.emit(
            "configured",
            sync_phases=sorted(self.sync_phases),
            warning="Diagnostic run: do not compare TPOT with a normal run.",
        )

    def emit(self, kind, **fields):
        with self.lock:
            self.serial += 1
            row = {
                **self.rank,
                "seq": self.serial,
                "step": self.step,
                "wall_ns": time.time_ns(),
                "mono_ns": time.monotonic_ns(),
                "kind": kind,
                **fields,
            }
            data = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=lambda value: {"type": type(value).__name__},
                )
                + "\n"
            ).encode()
            while data:
                data = data[os.write(self.fd, data) :]
            return self.serial

    def start(self):
        if self.active:
            return
        import torch

        self.module = torch.get_device_module(self.device)
        self.acl = None
        self.context = ctypes.c_void_p()
        try:
            acl = ctypes.CDLL("libascendcl.so")
            acl.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            acl.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
            acl.aclrtQueryEventStatus.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            for name in (
                "aclrtGetCurrentContext",
                "aclrtSetCurrentContext",
                "aclrtQueryEventStatus",
            ):
                getattr(acl, name).restype = ctypes.c_int
            code = acl.aclrtGetCurrentContext(ctypes.byref(self.context))
            if code != 0 or not self.context.value:
                raise RuntimeError(f"aclrtGetCurrentContext returned {code}")
            self.acl = acl
        except (OSError, AttributeError, RuntimeError) as exc:
            self.emit("completion_observer_unavailable", error=str(exc))
        self.active = True
        self.emit(
            "started", device=str(self.device), completion_observer=self.acl is not None
        )
        threading.Thread(
            target=self._observe_safe, name="dspark-diag-observer", daemon=True
        ).start()

    def stream_description(self):
        stream = self.module.current_stream()
        # Do not access npu_stream: its getter can drain the task queue.
        return {"torch_stream_id": stream.stream_id, "device": str(stream.device)}

    def begin(self, stage, **fields):
        token = self.emit("host_begin", stage=stage, **fields)
        with self.lock:
            self.open_spans[token] = {"stage": stage, "step": self.step, "token": token}
        return token

    def end(self, token, **fields):
        with self.lock:
            entry = self.open_spans.pop(token, {})
        self.emit("host_end", **entry, **fields)

    def checkpoint(self, stage):
        if len(self.events) >= self.event_limit:
            if len(self.events) == self.event_limit:
                self.emit("event_limit", limit=self.event_limit)
                self.event_limit = -1
            return None
        self.emit("event_record_begin", stage=stage)
        event = self.module.Event()
        event.record()
        entry = {
            "event": event,
            "stage": stage,
            "step": self.step,
            "complete": False,
            **self.stream_description(),
        }
        entry["event_id"] = self.emit(
            "device_submitted",
            **{k: v for k, v in entry.items() if k not in ("event", "complete")},
        )
        with self.lock:
            self.events.append(entry)
        if stage in self.sync_phases:
            self.synchronize(entry, stage)
        return entry

    def synchronize(self, entry, stage):
        token = self.begin("diagnostic_sync:" + stage, event_id=entry["event_id"])
        entry["event"].synchronize()
        self.emit("diagnostic_sync_complete", event_id=entry["event_id"], stage=stage)
        self.end(token)

    def graph_reuse(self, runner, role):
        previous = self.graph_events.get(id(runner))
        self.emit(
            "graph_reuse",
            role=role,
            runner=id(runner),
            previous_event_id=previous["event_id"] if previous else None,
            previous_complete_observed=previous["complete"] if previous else None,
        )
        if previous is not None and role + "_reuse" in self.sync_phases:
            self.synchronize(previous, role + "_reuse")

    def graph_submitted(self, runner, role, event):
        if event is not None:
            self.graph_events[id(runner)] = event
        buffers = runner.buffers
        names = (
            "input_ids",
            "seq_lens",
            "seq_lens_cpu",
            "req_pool_indices",
            "out_cache_loc",
            "global_num_tokens_gpu",
            "num_token_non_padded",
        )
        self.emit(
            "graph_inputs",
            role=role,
            runner=id(runner),
            key=str(runner._make_graph_key(runner.bs)),
            raw_bs=runner.raw_bs,
            captured_width=runner.captured_req_width,
            buffers={
                name: tensor_description(getattr(buffers, name, None)) for name in names
            },
        )

    def _observe_safe(self):
        try:
            self._observe()
        except Exception as exc:
            self.emit("observer_error", error_type=type(exc).__name__, error=str(exc))

    def _observe(self):
        acl = self.acl
        if acl is not None:
            code = acl.aclrtSetCurrentContext(self.context)
            if code != 0:
                self.emit(
                    "completion_observer_unavailable", error=f"set context: {code}"
                )
                acl = None
        next_heartbeat = 0.0
        while self.active:
            with self.lock:
                pending = [entry for entry in self.events if not entry["complete"]]
            if acl is not None:
                for entry in pending:
                    # Unique, never re-recorded events stay alive in self.events.
                    # A zero handle is not evidence of completion.
                    handle = entry["event"].npu_event
                    if not handle:
                        continue
                    status = ctypes.c_int(0)
                    code = acl.aclrtQueryEventStatus(
                        ctypes.c_void_p(handle), ctypes.byref(status)
                    )
                    if code != 0:
                        self.emit(
                            "device_query_error",
                            event_id=entry["event_id"],
                            stage=entry["stage"],
                            acl_error=code,
                        )
                        acl = None
                        break
                    # aclrtEventRecordedStatus: NOT_READY=0, COMPLETE=1.
                    if status.value == 1:
                        with self.lock:
                            entry["complete"] = True
                            self.last_complete[entry["stage"]] = entry["step"]
                        self.emit(
                            "device_complete_observed",
                            event_id=entry["event_id"],
                            stage=entry["stage"],
                            producer_step=entry["step"],
                        )
            if time.monotonic() >= next_heartbeat:
                frame = sys._current_frames().get(self.main_thread)
                stack = (
                    []
                    if frame is None
                    else [
                        {"file": f.filename, "line": f.lineno, "function": f.name}
                        for f in traceback.extract_stack(frame)[-12:]
                    ]
                )
                with self.lock:
                    spans = list(self.open_spans.values())
                    completed = dict(self.last_complete)
                self.emit(
                    "heartbeat",
                    open_spans=spans,
                    main_stack=stack,
                    last_complete=completed,
                    pending_events=len(pending),
                )
                next_heartbeat = time.monotonic() + 5
            time.sleep(0.05)
