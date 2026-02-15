#!/usr/bin/env bash
set -euo pipefail

screen -S backend -X quit >/dev/null 2>&1 || true
screen -S collector -X quit >/dev/null 2>&1 || true
screen -S trainer -X quit >/dev/null 2>&1 || true

echo "[screen sessions]"
screen -ls || true
