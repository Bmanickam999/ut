#!/usr/bin/env bash
# Recreate LLMDemo's local virtual environment safely on macOS or Linux.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${LLMDEMO_VENV:-${PROJECT_DIR}/.venv}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${VENV_DIR}.previous-${TIMESTAMP}"
FAILED_DIR="${VENV_DIR}.failed-${TIMESTAMP}"

usage() {
  cat <<'EOF'
Usage: ./recreate_venv.sh [--auto|--cpu|--gpu] [--run-tests] [--with-shampoo]

Safely recreates .venv by:
  1. Renaming an existing environment to .venv.previous-TIMESTAMP.
  2. Calling the project's canonical setup.sh installer.
  3. Restoring the previous environment automatically if setup fails.

Common office-laptop commands:
  ./recreate_venv.sh --cpu --run-tests
  ./recreate_venv.sh --gpu --run-tests

The script honors settings already supported by setup.sh:
  PYTHON_BIN         Explicit Python 3.10+ executable.
  LLMDEMO_VENV       Environment path; default is PROJECT/.venv.
  PYTORCH_INDEX_URL  Official PyTorch wheel index, for example cu128.

Standard pip corporate-network variables are inherited automatically:
  HTTPS_PROXY, HTTP_PROXY, NO_PROXY
  PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, PIP_CERT, PIP_CLIENT_CERT

Run from macOS, Linux, or WSL. Native Windows PowerShell should use WSL or
perform the equivalent Python venv commands described in README.md.
EOF
}

ARGS=()
while (($#)); do
  case "$1" in
    --auto|--cpu|--gpu|--run-tests|--with-shampoo) ARGS+=("$1") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" || ! -f "${PROJECT_DIR}/setup.sh" ]]; then
  echo "Refusing to continue: ${PROJECT_DIR} is not a complete LLMDemo checkout." >&2
  exit 1
fi

case "${VENV_DIR}" in
  ""|/|"${HOME}"|"${PROJECT_DIR}")
    echo "Refusing unsafe virtual-environment path: ${VENV_DIR}" >&2
    exit 1
    ;;
esac

# Only the exact configured environment is moved; no broad path or glob is
# used. This makes the operation recoverable and avoids deleting user data.
HAD_PREVIOUS=0
if [[ -e "${VENV_DIR}" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Refusing to replace non-directory path: ${VENV_DIR}" >&2
    exit 1
  fi
  echo "Preserving existing environment as: ${BACKUP_DIR}"
  mv "${VENV_DIR}" "${BACKUP_DIR}"
  HAD_PREVIOUS=1
fi

restore_previous() {
  status=$?
  if ((status == 0)); then
    return
  fi
  echo "Environment setup failed with status ${status}." >&2
  if [[ -d "${VENV_DIR}" ]]; then
    mv "${VENV_DIR}" "${FAILED_DIR}"
    echo "Incomplete environment preserved as: ${FAILED_DIR}" >&2
  fi
  if ((HAD_PREVIOUS)) && [[ -d "${BACKUP_DIR}" ]]; then
    mv "${BACKUP_DIR}" "${VENV_DIR}"
    echo "Previous environment restored: ${VENV_DIR}" >&2
  fi
  exit "${status}"
}
trap restore_previous EXIT

echo "Recreating ${VENV_DIR} using setup.sh ${ARGS[*]:---auto}"
if ((${#ARGS[@]})); then
  bash "${PROJECT_DIR}/setup.sh" "${ARGS[@]}"
else
  bash "${PROJECT_DIR}/setup.sh" --auto
fi

trap - EXIT

cat <<EOF

Virtual environment recreated successfully.

Activate it:
  source "${VENV_DIR}/bin/activate"

Verify the installed CLI:
  "${VENV_DIR}/bin/llmdemo" doctor

Run the CPU smoke demo:
  "${PROJECT_DIR}/run_demo.sh" --smoke --cpu
EOF

if ((HAD_PREVIOUS)); then
  cat <<EOF

The previous environment remains available at:
  ${BACKUP_DIR}

After validating the new environment, you may delete that backup manually.
EOF
fi
