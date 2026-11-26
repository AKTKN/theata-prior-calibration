#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_RUN="${PYTHON_RUN:-${PROJECT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_RUN}" ]]; then
  echo "Python environment not found. Run 'bash scripts/bootstrap.sh' or set PYTHON_RUN." >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cd -- "${PROJECT_DIR}"
if [[ "${1:-}" == structure ]]; then
  exec "${PYTHON_RUN}" -m theata_prior "$@"
fi
exec "${PYTHON_RUN}" -m theata_prior run "$@"

