#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-all}" # runtime|train|dev|all
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
ALLOW_NON_312="${ALLOW_NON_312:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[FAIL] python executable not found: $PYTHON_BIN"
    exit 2
  fi
fi

PY_VER="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$PY_VER" != "3.12" && "$ALLOW_NON_312" != "1" ]]; then
  echo "[FAIL] python version must be 3.12.x for reproducible runtime. got: $PY_VER"
  echo "set PYTHON_BIN=python3.12 or set ALLOW_NON_312=1 to bypass"
  exit 2
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

install_runtime() {
  python -m pip install -r requirements-runtime.txt
}

install_train() {
  python -m pip install -r requirements-train.txt
}

install_dev() {
  python -m pip install -r requirements-dev.txt
}

case "$PROFILE" in
  runtime)
    install_runtime
    ;;
  train)
    install_runtime
    install_train
    ;;
  dev)
    install_runtime
    install_dev
    ;;
  all)
    install_runtime
    install_train
    install_dev
    ;;
  *)
    echo "[FAIL] unsupported PROFILE=${PROFILE} (runtime|train|dev|all)"
    exit 2
    ;;
esac

echo "[env]"
echo "python=$(command -v python)"
echo "pip=$(command -v pip)"
python -V
python -m pytest --version
pytest --version
echo "[OK] bootstrap completed"
