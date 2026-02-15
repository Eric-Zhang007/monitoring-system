#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/4] create venv: ${VENV_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "[2/4] upgrade pip"
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] install backend deps"
python -m pip install -r "${ROOT_DIR}/backend/requirements.txt"

echo "[4/4] run tests"
PYTHONPATH="${ROOT_DIR}/backend" python -m pytest -q "${ROOT_DIR}/backend/tests"

echo "dev tests passed"
