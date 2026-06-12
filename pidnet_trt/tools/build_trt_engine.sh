#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 model.onnx [model.fp16.trt]" >&2
  exit 2
fi

onnx_path="$1"
engine_path="${2:-${onnx_path%.onnx}.fp16.trt}"
workspace_mb="${TRT_WORKSPACE_MB:-2048}"
precision="${TRT_PRECISION:-fp16}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec was not found in PATH. Run this where TensorRT is installed." >&2
  exit 1
fi

if [ ! -f "${onnx_path}" ]; then
  echo "ONNX file not found: ${onnx_path}" >&2
  exit 1
fi

args=(
  "--onnx=${onnx_path}"
  "--saveEngine=${engine_path}"
  "--memPoolSize=workspace:${workspace_mb}"
)

case "${precision}" in
  fp32)
    ;;
  fp16)
    args+=("--fp16")
    ;;
  *)
    echo "Unsupported TRT_PRECISION=${precision}; use fp16 or fp32." >&2
    exit 1
    ;;
esac

# shellcheck disable=SC2086
trtexec "${args[@]}" ${TRTEXEC_EXTRA_ARGS:-}

echo "Wrote ${engine_path}"

