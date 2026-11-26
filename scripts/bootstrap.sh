#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
THEATA_DIR="${THEATA_PATH:-${PROJECT_DIR}/vendor/theata}"
THEATA_REV="8bfb9b97de91bacac2f56725c27b4c71c6a91009"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
if [[ ! -d "${THEATA_DIR}" ]]; then
  mkdir -p -- "$(dirname -- "${THEATA_DIR}")"
  git clone --filter=blob:none https://github.com/MSTakahashiPJ/theata.git "${THEATA_DIR}"
  git -C "${THEATA_DIR}" checkout --detach "${THEATA_REV}"
fi
if [[ "$(git -C "${THEATA_DIR}" rev-parse HEAD)" != "${THEATA_REV}" ]]; then
  echo "THeatA checkout is not the tested revision ${THEATA_REV}. Use a separate checkout at that revision." >&2
  exit 2
fi
"${PYTHON_BOOTSTRAP}" -m venv "${PROJECT_DIR}/.venv"
"${PROJECT_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${PROJECT_DIR}/.venv/bin/python" -m pip install -e "${PROJECT_DIR}[notebook,test]"
"${PROJECT_DIR}/.venv/bin/python" -m ipykernel install --prefix "${PROJECT_DIR}/.venv" --name theata-prior --display-name "THeatA prior calibration"
echo "Setup complete. Run: bash scripts/run.sh --config configs/smoke.yaml --out results/smoke --workers 2"

