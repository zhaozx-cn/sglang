# Kimi-K3 DSPark draft prefetch on Ascend NPU

This port adds opt-in draft prefetch to the speculative overlap path. Kimi-K3
DSPark uses ACLGraph for both Target and Draft. The optional Tensor FIA binding
lets the dense Draft consume device prefix lengths without a replay-time D2H
or a host FIA attribute update. The DSPark-specific TorchAir GE backend has
been removed; unrelated framework `torch.compile` support is unchanged.

The A5 integration is experimental: the external A5 kernel has been
cross-compiled, but A5 operator accuracy, serving stability, acceptance length
and performance have **not** been validated on hardware. Do not interpret
this framework update as a proven A5 speedup or precision guarantee.

## Dependencies for A5

Bring both the SGLang changes in this PR and the separate
`ops-transformer-fia-device-seq` source containing `prototype/a5`. The latter
is **not bundled in this SGLang PR** and is not a stock CANN FIA entry point.
Build it in the target container for its exact Ascend950 SoC, CANN version
and PyTorch C++ ABI. An A3 device binary must not be used on A5, nor should a
binding built against a different container ABI be reused.

From that external source root (substitute the actual SoC name):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S prototype/a5 -B prototype/build-a5 -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend950PR_9599 -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" -DTORCH_SITE="$(python3 -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent.parent)')" -DTORCH_CXX11_ABI="$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
cmake --build prototype/build-a5 -j4
```

The adapter loads `libk3_fia_tensor.so`, checks
`torch.ops.k3_fia_experimental.abi_version() == 2`, and prepares a cached native
plan per graph shape during warmup. Replay passes the original int32/int64 NPU
prefix tensor to `run`; the native device task handles lengths. It does not
add a separate planning task or stream. Device planning still has execution
cost and requires A5 measurement. Workspace addresses remain stable after
warmup; unsupported growth during capture/replay is rejected.

A working, version-matched `sgl-kernel-npu` is still needed to run the K3
baseline. This Tensor FIA route does **not** require our earlier experimental
`verify_gqa` or GE/Triton compatibility modifications in that repository.
It does not mean the model's existing kernel package can be uninstalled.

## Serving configuration

Keep the existing working Kimi-K3 DSPark model paths, TP/DP topology, memory
fraction and Radix cache configuration. Apply these variables inside every
serving container, before launching Python, after validating the native A5
artifacts:

```bash
export ASCEND_USE_FIA=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FIA_TENSOR=1
export SGLANG_DSPARK_FIA_TENSOR_LIB=/path/to/ops-transformer-fia-device-seq/prototype/build-a5/lib/libk3_fia_tensor.so
export SGLANG_DSPARK_FIA_TENSOR_BINARY=/path/to/ops-transformer-fia-device-seq/prototype/build-a5/k3_fia_a5_kernel_merge_obj_dir/device.o
export SGLANG_NPU_USE_FIAS_V2_BSND=0
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_DSPARK_TARGET_GRAPH_REUSE_GUARD=1
export SGLANG_DSPARK_DEFER_TARGET_METADATA=0
export SGLANG_DSPARK_DIAG_DIR=
export SGLANG_DSPARK_DIAG_SYNC_PHASES=
export SGLANG_LOG_DECODE_GRAPH_KEY=0
unset SGLANG_DSPARK_TORCHAIR_FIA ASCEND_LAUNCH_BLOCKING
```

The supported Tensor FIA shape is deliberately bounded: non-causal dense
attention without sinks, BF16 Q/K/V, TND Q `[batch*7,4,64]`, ND paged K/V
`[pages,128,64]`, page size 128, local batch 1..32, and attention-TP=16 for the
K3 Draft's 64 Q heads / 16 KV heads. Padded graph rows retain zero prefix
length; the native kernel adds seven slots only to positive prefixes.
Do not change TP/DP merely to enable this feature; use a compatible layout.

Add or align the following arguments in the existing working launch command
(this is an argument list, not a standalone shell command):

```text
--enable-draft-prefetch
--device npu
--attention-backend ascend
--speculative-algorithm DSPARK
--speculative-dspark-block-size 7
--speculative-eagle-topk 1
--speculative-draft-attention-backend ascend
--speculative-draft-model-quantization unquant
--speculative-draft-kv-cache-dtype bfloat16
--linear-attn-verify-backend triton
--page-size 128
```

Keep graph capture and overlap scheduling enabled, and `torch.compile`
disabled for this path. Draft graph buckets must not exceed 32. Keep existing
valid Target/Draft buckets aligned between A/B runs rather than adding new
ones just for prefetch. Eager Draft overflow above 32 rows uses exact CPU
lengths and therefore **does incur D2H**; it is not the device-only fast path.
This limit is local to each attention-DP worker, not global client concurrency.

The Tensor FIA switch defaults to 0. Without it the original ACLGraph + host
FIA metadata path remains available, including functional prefetch with CPU
length synchronization. The removed `SGLANG_DSPARK_TORCHAIR_FIA` switch no longer
enables GE or substitutes for `SGLANG_DSPARK_FIA_TENSOR`.

Do **not** use `--skip-draft-prefetch-seq-lens-cpu-sync` to bypass K3 Target
metadata requirements. Tensor FIA can consume Draft device lengths; Target
FIA still uses an exact pinned CPU length snapshot. The NPU path issues that
snapshot on the existing private D2H stream after accept and waits only for
its completion event when the next Target requires it.

The steady-state prefetch path supports static greedy verification. Bootstrap,
new/mixed requests and non-greedy cases retain proposer/fallback handling.
The draft is submitted on the existing forward stream, not a background
draft thread or an additional draft compute stream.

## Reuse protection is not diagnostic instrumentation

`SGLANG_DSPARK_TARGET_GRAPH_REUSE_GUARD=1` is independent of diagnostics. For NPU
DSPark with prefetch enabled, every Target runner retains one completion event,
including idle DP participation. Before loading the next graph's input slots
or updating host attributes it waits for the previous execution. The event is
re-recorded only after that wait. Graph bucket changes share the same guard.

The diagnostic `target_reuse` experiment allowed the full model to complete
requests where an unprotected run stalled and later reported a
`MoeLowLatencyDispatchV2` failure. This supports a conservative reuse fence;
it does not prove which internal resource or ordering caused that failure.
The fence's synchronization cost is part of serving latency, not a zero-cost
performance claim. Draft and non-prefetch runners do not use this guard.

Leave diagnostic flags empty for latency measurements. See
[DSPark diagnostics](dspark_diagnostics.md) for isolated debugging. Cluster IPs,
container-specific launchers, credentials, raw logs, weights, traces and kernel
dumps are intentionally not included in this contribution.

## Rank0 decode profiling

Start the service with `SGLANG_PROFILE_V2=0` and `SGLANG_PROFILE_RANKS=0`, then:

```bash
SERVER_URL=http://127.0.0.1:15010 PROFILE_STEPS=5 ./profile_kimi_k3_decode_rank0.sh start
```

Send the workload after arming the profiler. All ranks retain the same
scheduler/collective behavior, while only the selected global TP rank creates
the profiler. Do not compare instrumented request latency with a normal
benchmark. Exclude capture/startup and profiler-stop boundaries when measuring
steady-state device steps.

## Validation before A5 serving

The external A5 source currently has Ascend950 cross-compilation and 160,000
CPU native/device-plan comparisons under ASan/UBSan. These checks do not run
the A5 attention kernel and do not establish numerical or performance parity.
Previous A3 operator checks also do not establish A5 correctness.

On an otherwise idle A5 device, run the external validation harness first:

```bash
python3 prototype/check_npu.py --device 0 --reference-api v2 --strict --library "$PWD/prototype/build-a5/lib/libk3_fia_tensor.so" --binary "$PWD/prototype/build-a5/k3_fia_a5_kernel_merge_obj_dir/device.o" --adapter /path/to/sglang/python/sglang/srt/hardware_backend/npu/attention/dspark_tensor_fia.py --benchmark-iters 100
```

This checks heterogeneous lengths, padding, both prefix dtypes and changing
lengths at fixed addresses across ACLGraph replays. Do not relax `--strict`
to hide discrepancies. Follow with multi-request serving and controlled
acceptance/TPOT comparisons without profiling, using identical weights,
requests, sampling, seed, cache state and launch configuration. No A5 serving
results are claimed in this PR; old GE profile timings do not describe this
ACLGraph Tensor FIA implementation.

The existing merge of target-branch changes through `6b712e962`, including
the persistent NPU graph-update worker, is retained. This update does not
implement DCP, change native attention math in SGLang, or restart any service.
