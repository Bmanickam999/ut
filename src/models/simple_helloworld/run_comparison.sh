#!/usr/bin/env bash
# Run the preserved lightweight coupled and decoupled demonstrations.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./setup.sh --cpu first." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="artifacts/simple-helloworld-comparison/${RUN_ID}"
export MPLCONFIGDIR="${RUN_DIR}/.matplotlib"

LLMDEMO_DEVICE=cpu .venv/bin/python -m models.simple_helloworld.before.helloworld \
  --output-dir "${RUN_DIR}/before"

LLMDEMO_DEVICE=cpu .venv/bin/python -m generic_trainer.cli \
  --config src/models/simple_helloworld/new/config.yaml \
  --set "run.output_dir=${RUN_DIR}/after"

.venv/bin/llmdemo compare \
  --before "${RUN_DIR}/before" \
  --after "${RUN_DIR}/after" \
  --output "${RUN_DIR}/comparison" \
  --before-label "Simple Before" \
  --after-label "Simple After"

echo "Simple comparison: ${RUN_DIR}/comparison/run_comparison.png"
