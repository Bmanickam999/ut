#!/usr/bin/env bash
# Run the smoke profile or the 500M profile after bootstrapping dependencies.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${LLMDEMO_VENV:-${PROJECT_DIR}/.venv}"
PYTHON_COMMAND="${VENV_DIR}/bin/python"
PROFILE="smoke"
DEVICE="auto"

while (($#)); do
  case "$1" in
    --smoke) PROFILE="smoke" ;;
    --500m) PROFILE="train_500m" ;;
    --1b) PROFILE="train_1b" ;;
    --cpu) DEVICE="cpu" ;;
    --gpu|--cuda) DEVICE="cuda" ;;
    --mps) DEVICE="mps" ;;
    -h|--help)
      echo "Usage: NUM_GPUS=N ./run_demo.sh [--smoke|--500m|--1b] [--cpu|--gpu|--mps]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -x "${PYTHON_COMMAND}" ]]; then
  echo "No virtual environment found; running automatic setup first."
  "${PROJECT_DIR}/setup.sh" --auto
fi

cd "${PROJECT_DIR}"
GPU_COUNT="${NUM_GPUS:-1}"
if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer." >&2
  exit 2
fi
if [[ "${DEVICE}" != "cuda" && "${GPU_COUNT}" -gt 1 ]]; then
  echo "NUM_GPUS > 1 requires --gpu." >&2
  exit 2
fi
export LLMDEMO_DEVICE="${DEVICE}"

if ((GPU_COUNT > 1)); then
  exec "${VENV_DIR}/bin/torchrun" --standalone --nproc_per_node="${GPU_COUNT}" \
    -m generic_trainer.cli --config "configs/${PROFILE}.yaml"
else
  exec "${PYTHON_COMMAND}" -m generic_trainer.cli --config "configs/${PROFILE}.yaml"
fi
