# DSPark hang diagnostics

Normal startup leaves `SGLANG_DSPARK_DIAG_DIR` empty. Diagnostic decorators then
return the original functions; no event observer or diagnostic events exist.
These changes do not change model math, sampling, TP/DP, graph tiers or weights.

Normal NPU DSPark **prefetch** serving now retains the conservative Target graph
reuse fence independently via `SGLANG_DSPARK_TARGET_GRAPH_REUSE_GUARD=1` (default).
Each Target runner records one reusable completion event on its existing forward
stream and synchronizes it before its next input load/host graph update, including
idle DP participation and changes of graph bucket. There is no diagnostic
observer or extra stream. Draft runners and non-prefetch runners do not use this
fence. Its actual synchronization cost remains part of serving TPOT.

For a normal performance run, leave `SGLANG_DSPARK_DIAG_DIR` and
`SGLANG_DSPARK_DIAG_SYNC_PHASES` empty and set `SGLANG_LOG_DECODE_GRAPH_KEY=0`.
Do not disable the reuse guard just to remove diagnostic overhead. When repeating
the old `target_reuse` diagnostic A/B, explicitly set the independent guard to
`0` on **all four nodes**; otherwise the production fence also protects the
nominal diagnostic-off arm and the comparison no longer isolates that wait.

Set `SGLANG_DSPARK_DIAG_DIR` to a **new absolute directory per run** on every
node through your launch environment. Each rank writes `rankNNN_pidPID.jsonl`
locally, without depending on HTTP, Gloo or scheduler IPC. Copy the three remote
directories back to a collection directory to obtain all 64 ranks.

Default diagnostic mode records without adding waits. Host entry/exit covers
scheduling, Gloo, KV allocation/H2D, sequence-length publish/resolve, graph
input preparation/update/replay, accept, both commit stages, draft proposal and
prefetch, WAR and result processing. Idle DP Target forwards are also recorded.
Each device checkpoint has a unique event recorded on the **existing** stream.
No diagnostic compute stream or collective is created. A CPU thread polls
`aclrtQueryEventStatus` in the existing context, bypassing torch_npu's dispatch
queue, and writes progress and the main Python stack every five seconds.
Errors/unavailable completion observation are explicit, not treated as success.

Optional `SGLANG_DSPARK_DIAG_SYNC_PHASES` is a comma-separated subset of:

- `target_graph`, `draft_graph`: synchronize immediately after that graph.
- `target_reuse`, `draft_reuse`: synchronize the previous execution before the
  same runner's input slots are rewritten (including changes of graph bucket).
- `accept`, `commit_kda`, `commit_hidden`: isolate post-verify state production.
- `draft_propose`, `draft_prefetch`, `worker_forward`, `seq_lens_d2h`:
  isolate the corresponding stage on its existing stream.

Use the same flags on all four nodes. These are **diagnostic isolation waits**,
not the proposed production fix. Do not use a diagnostic run for TPOT results.
No API changes the mode dynamically while a distributed forward is in flight.

Summarize collected logs with:

```bash
python3 tools/summarize_dspark_diagnostics.py /absolute/collected/diagnostics
python3 tools/summarize_dspark_diagnostics.py /absolute/collected/diagnostics --json
```

`device_submitted` means the host recorded the checkpoint, not that the device
finished. `device_complete_observed` is an observed completion of that stream's
prefix, not an exact NPU timestamp or a particular kernel's duration. A pending
previous graph at reuse is **not** by itself a race: same-stream ordering or an
existing wait can protect it. Compare all ranks and the WAR/dependency records.
Events are retained, never re-recorded, capped at 8192 per process; reaching the
cap explicitly disables new checkpoints. Capture is not instrumented: the
observer starts only when the scheduler enters serving, before HTTP warmup.
