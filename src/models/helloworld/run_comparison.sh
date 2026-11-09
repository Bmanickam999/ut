#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_DIR}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./setup.sh --cpu first." >&2
  exit 1
fi

NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="artifacts/helloworld-comparison/${RUN_ID}"
export MPLCONFIGDIR="${RUN_DIR}/.matplotlib"

echo "Comparison run: ${RUN_DIR}"
echo "Examples: ${NUM_EXAMPLES}; epochs: ${EPOCHS}; batch size: ${BATCH_SIZE}"
echo "=== BEFORE: coupled model + infrastructure ==="
LLMDEMO_DEVICE=cpu .venv/bin/python -m models.helloworld.before.helloworld \
  --epochs "${EPOCHS}" --examples "${NUM_EXAMPLES}" --batch-size "${BATCH_SIZE}" \
  --output-dir "${RUN_DIR}/before"

echo "=== AFTER: model-only code + generic trainer ==="
LLMDEMO_DEVICE=cpu .venv/bin/python -m generic_trainer.cli \
  --config src/models/helloworld/new/config.yaml \
  --set "run.output_dir=${RUN_DIR}/after" \
  --set "data.num_examples=${NUM_EXAMPLES}" \
  --set "trainer.max_epochs=${EPOCHS}" \
  --set "trainer.train_batch_size=${BATCH_SIZE}" \
  --set "trainer.eval_batch_size=${BATCH_SIZE}"

echo "=== VISUAL COMPARISON ==="
.venv/bin/python -m models.helloworld.compare \
  --before "${RUN_DIR}/before" --after "${RUN_DIR}/after" \
  --output "${RUN_DIR}/visuals"

echo "Dashboard: ${RUN_DIR}/visuals/leadership_comparison.png"
echo "TensorBoard: .venv/bin/tensorboard --logdir ${RUN_DIR} --port 6006"
