#!/usr/bin/env bash
# Generic wrapper for comparing two standard trainer artifact directories.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 BEFORE_DIR AFTER_DIR OUTPUT_DIR [BEFORE_LABEL] [AFTER_LABEL]" >&2
  exit 2
fi

BEFORE_DIR="$1"
AFTER_DIR="$2"
OUTPUT_DIR="$3"
BEFORE_LABEL="${4:-Before}"
AFTER_LABEL="${5:-After}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

.venv/bin/python -m generic_trainer.comparison \
  --before "${BEFORE_DIR}" \
  --after "${AFTER_DIR}" \
  --output "${OUTPUT_DIR}" \
  --before-label "${BEFORE_LABEL}" \
  --after-label "${AFTER_LABEL}"
