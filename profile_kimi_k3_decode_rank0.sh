#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-start}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:15010}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-$(pwd)/profiling/kimi_k3_decode_rank0_${RUN_TAG}}"

if ! [[ "${PROFILE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_STEPS must be a positive integer, got: ${PROFILE_STEPS}" >&2
    exit 2
fi

case "${PROFILE_RECORD_SHAPES}" in
    0) PROFILE_RECORD_SHAPES_JSON=false ;;
    1) PROFILE_RECORD_SHAPES_JSON=true ;;
    *)
        echo "PROFILE_RECORD_SHAPES must be 0 or 1, got: ${PROFILE_RECORD_SHAPES}" >&2
        exit 2
        ;;
esac

case "${ACTION}" in
    start)
        mkdir -p "${PROFILE_OUTPUT_DIR}"
        curl --fail --silent --show-error \
            --request POST \
            --header "Content-Type: application/json" \
            --data-binary @- \
            "${SERVER_URL%/}/start_profile" <<JSON
{
  "output_dir": "${PROFILE_OUTPUT_DIR}",
  "num_steps": ${PROFILE_STEPS},
  "activities": ["CPU", "GPU"],
  "profile_by_stage": true,
  "profile_stages": ["decode"],
  "with_stack": false,
  "record_shapes": ${PROFILE_RECORD_SHAPES_JSON},
  "merge_profiles": false,
  "profile_prefix": "kimi-k3-decode-rank0",
  "profile_id": "${RUN_TAG}"
}
JSON
        echo
        echo "Profiling armed: decode=${PROFILE_STEPS} steps, global TP rank=0 only"
        echo "Record shapes: ${PROFILE_RECORD_SHAPES}"
        echo "Trace directory: ${PROFILE_OUTPUT_DIR}"
        echo "Send a workload with at least ${PROFILE_STEPS} decode batches now."
        ;;
    stop)
        curl --fail --silent --show-error \
            --request POST \
            "${SERVER_URL%/}/stop_profile"
        echo
        echo "Profiling stopped."
        ;;
    *)
        echo "Usage: SERVER_URL=http://host:port $0 {start|stop}" >&2
        exit 2
        ;;
esac
