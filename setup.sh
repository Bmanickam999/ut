#!/usr/bin/env bash
# Portable, idempotent bootstrap for macOS and Linux, CPU or NVIDIA GPU.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${LLMDEMO_VENV:-${PROJECT_DIR}/.venv}"
MODE="auto"
RUN_TESTS=0
WITH_SHAMPOO=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--auto|--cpu|--gpu] [--with-shampoo] [--run-tests]

  --auto       Select NVIDIA GPU when nvidia-smi is available; otherwise CPU.
  --cpu        Install a CPU build of PyTorch where a dedicated wheel exists.
  --gpu        Install PyTorch and require a working CUDA runtime afterward.
  --run-tests  Run the full test suite after installation.
  --with-shampoo  Install Meta's optional Distributed Shampoo package.

Environment variables:
  PYTHON_BIN         Python 3.10+ executable (default: python3, then python)
  LLMDEMO_VENV       Virtual-environment directory (default: PROJECT/.venv)
  PYTORCH_INDEX_URL  Optional official PyTorch wheel index for a specific CUDA
                     platform, e.g. https://download.pytorch.org/whl/cu128
EOF
}

while (($#)); do
  case "$1" in
    --auto) MODE="auto" ;;
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --run-tests) RUN_TESTS=1 ;;
    --with-shampoo) WITH_SHAMPOO=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_COMMAND="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_COMMAND="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_COMMAND="python"
else
  echo "Python 3.10+ is required but was not found." >&2
  exit 1
fi

"${PYTHON_COMMAND}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required; found {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ "${MODE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    MODE="gpu"
  else
    MODE="cpu"
  fi
fi

echo "Creating isolated environment: ${VENV_DIR}"
"${PYTHON_COMMAND}" -m venv "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Could not create ${VENV_PYTHON}. Ensure the Python venv module is installed." >&2
  exit 1
fi

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

OS_NAME="$(uname -s)"
if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
  echo "Installing PyTorch from ${PYTORCH_INDEX_URL}"
  "${VENV_PYTHON}" -m pip install torch --index-url "${PYTORCH_INDEX_URL}"
elif [[ "${MODE}" == "cpu" && "${OS_NAME}" == "Linux" ]]; then
  echo "Installing the official CPU-only PyTorch wheel"
  "${VENV_PYTHON}" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  # PyPI selects the supported macOS wheel or the standard Linux CUDA wheel.
  echo "Installing PyTorch from PyPI for mode=${MODE}"
  "${VENV_PYTHON}" -m pip install torch
fi

# --no-deps prevents pip from replacing the already selected hardware-specific
# torch wheel. Install the remaining runtime and notebook/test dependencies.
"${VENV_PYTHON}" -m pip install PyYAML pytest jupyter matplotlib tensorboard
"${VENV_PYTHON}" -m pip install --no-deps -e "${PROJECT_DIR}"

if ((WITH_SHAMPOO)); then
  echo "Installing optional Distributed Shampoo from Meta's official repository"
  "${VENV_PYTHON}" -m pip install "git+https://github.com/facebookresearch/optimizers.git"
fi

"${VENV_PYTHON}" - "${MODE}" <<'PY'
import sys
import torch
import yaml

mode = sys.argv[1]
print(f"PyTorch: {torch.__version__}")
print(f"PyYAML: {yaml.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Gloo available: {torch.distributed.is_gloo_available()}")
print(f"NCCL available: {torch.distributed.is_nccl_available()}")
if mode == "gpu" and not torch.cuda.is_available():
    raise SystemExit(
        "GPU mode was requested but CUDA is unavailable. Check the NVIDIA driver "
        "and choose the matching official wheel with PYTORCH_INDEX_URL."
    )
if mode == "gpu" and not torch.distributed.is_nccl_available():
    raise SystemExit("GPU DDP requires a PyTorch build with NCCL support.")
print("Dependency verification passed.")
PY

if ((RUN_TESTS)); then
  "${VENV_PYTHON}" -m pytest -q "${PROJECT_DIR}/tests" -rs
fi

cat <<EOF

LLMDemo is ready.
Activate: source "${VENV_DIR}/bin/activate"
CPU smoke test: ./run_demo.sh --smoke --cpu
Apple GPU smoke test: ./run_demo.sh --smoke --mps
Multi-GPU 500M: NUM_GPUS=4 ./run_demo.sh --500m --gpu
EOF
