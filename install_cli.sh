#!/usr/bin/env bash
# Install the llmdemo CLI into an isolated user environment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${LLMDEMO_CLI_HOME:-${HOME}/.local/share/llmdemo}"
BIN_DIR="${LLMDEMO_BIN_DIR:-${HOME}/.local/bin}"
MODE="auto"
WITH_SHAMPOO=0

usage() {
  cat <<'EOF'
Usage: ./install_cli.sh [--auto|--cpu|--gpu] [--with-shampoo]

Installs an isolated CLI under ~/.local/share/llmdemo and creates
~/.local/bin/llmdemo. Override those paths with LLMDEMO_CLI_HOME and
LLMDEMO_BIN_DIR.
EOF
}

while (($#)); do
  case "$1" in
    --auto) MODE="auto" ;;
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --with-shampoo) WITH_SHAMPOO=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PYTHON_COMMAND="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "${PYTHON_COMMAND}" ]]; then
  echo "Python 3.10+ was not found." >&2
  exit 1
fi
"${PYTHON_COMMAND}" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'

if [[ "${MODE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then MODE="gpu"; else MODE="cpu"; fi
fi

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"
"${PYTHON_COMMAND}" -m venv "${INSTALL_ROOT}/venv"
CLI_PYTHON="${INSTALL_ROOT}/venv/bin/python"
"${CLI_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
  "${CLI_PYTHON}" -m pip install torch --index-url "${PYTORCH_INDEX_URL}"
elif [[ "${MODE}" == "cpu" && "$(uname -s)" == "Linux" ]]; then
  "${CLI_PYTHON}" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  "${CLI_PYTHON}" -m pip install torch
fi
"${CLI_PYTHON}" -m pip install PyYAML tensorboard
"${CLI_PYTHON}" -m pip install --no-deps --no-build-isolation "${PROJECT_DIR}"
if ((WITH_SHAMPOO)); then
  "${CLI_PYTHON}" -m pip install "git+https://github.com/facebookresearch/optimizers.git"
fi

ln -sfn "${INSTALL_ROOT}/venv/bin/llmdemo" "${BIN_DIR}/llmdemo"
"${BIN_DIR}/llmdemo" doctor

echo
echo "Installed: ${BIN_DIR}/llmdemo"
case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) echo "Add this to your shell profile: export PATH=\"${BIN_DIR}:\$PATH\"" ;;
esac
