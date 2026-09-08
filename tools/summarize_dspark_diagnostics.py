"""Summarize per-rank DSPark progress without contacting the service IPC path."""

import argparse
import json
from pathlib import Path


def summarize(path):
    submitted = {}
    completed = {}
    open_spans = {}
    last = {}
    heartbeat = {}
    errors = []
    graphs = {}
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A running process may be in the middle of its last write.
        last = row
        kind = row["kind"]
        if kind == "host_begin":
            open_spans[row["seq"]] = row["stage"]
        elif kind == "host_end":
            open_spans.pop(row.get("token"), None)
        elif kind == "device_submitted":
            submitted[row["stage"]] = row["step"]
        elif kind == "device_complete_observed":
            completed[row["stage"]] = row["producer_step"]
        elif kind == "graph_inputs":
            graphs[row["role"]] = {k: row[k] for k in ("step", "key", "raw_bs")}
        elif kind == "heartbeat":
            heartbeat = row
        elif kind in (
            "exception",
            "device_query_error",
            "observer_error",
            "completion_observer_unavailable",
            "event_limit",
        ):
            errors.append(row)
    return {
        "path": str(path),
        "rank": last.get("tp_rank"),
        "dp": last.get("dp_rank"),
        "pid": last.get("pid"),
        "step": last.get("step"),
        "last_kind": last.get("kind"),
        "last_wall_ns": last.get("wall_ns"),
        "submitted": submitted,
        "completed_observed": completed,
        "open_spans": list(open_spans.values()),
        "main_stack": heartbeat.get("main_stack", []),
        "graphs": graphs,
        "errors": errors[-3:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [
        summarize(path) for path in sorted(args.directory.rglob("rank*_pid*.jsonl"))
    ]
    if not rows:
        raise SystemExit(f"No rank diagnostics found in {args.directory}")
    for row in rows:
        if args.json:
            print(json.dumps(row, ensure_ascii=False))
            continue

        def progress(stage):
            return f"{row['submitted'].get(stage, '-')}/{row['completed_observed'].get(stage, '-')}"

        stack = row["main_stack"]
        location = "-" if not stack else f"{stack[-1]['function']}:{stack[-1]['line']}"
        print(
            f"TP{row['rank']} DP{row['dp']} pid={row['pid']} step={row['step']} "
            f"Target={progress('target_graph')} Draft={progress('draft_graph')} "
            f"at={location} open={','.join(row['open_spans'][-4:])} "
            f"errors={len(row['errors'])}"
        )
    if not args.json:
        print(
            "Target/Draft = last submitted step / last completion-observed step; '-' is UNKNOWN, not failure."
        )
        print(
            "Completion is polled (~50 ms), not a device timestamp. Pending reuse alone does not prove a race."
        )


if __name__ == "__main__":
    main()
