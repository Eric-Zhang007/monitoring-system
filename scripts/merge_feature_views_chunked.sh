#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

START="${START:-2018-01-01T00:00:00Z}"
END="${END:-2026-03-04T00:00:00Z}"
CHUNK_DAYS="${CHUNK_DAYS:-7}"
DO_TRUNCATE="${DO_TRUNCATE:-1}"

to_epoch() {
  "${PYTHON_BIN}" - <<'PY' "$1"
import sys
from datetime import datetime, timezone
txt=sys.argv[1].strip().replace(" ","T")
if txt.endswith("Z"):
    txt=txt[:-1]+"+00:00"
dt=datetime.fromisoformat(txt)
if dt.tzinfo is None:
    dt=dt.replace(tzinfo=timezone.utc)
print(int(dt.timestamp()))
PY
}

to_isoz() {
  "${PYTHON_BIN}" - <<'PY' "$1"
import sys
from datetime import datetime, timezone
ts=int(sys.argv[1])
print(datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00","Z"))
PY
}

start_epoch="$(to_epoch "${START}")"
end_epoch="$(to_epoch "${END}")"
if (( end_epoch <= start_epoch )); then
  echo "invalid_range start=${START} end=${END}" >&2
  exit 2
fi
step_sec=$(( CHUNK_DAYS * 86400 ))
if (( step_sec <= 0 )); then
  echo "invalid_chunk_days:${CHUNK_DAYS}" >&2
  exit 2
fi

first=1
idx=0
cur="${start_epoch}"
while (( cur < end_epoch )); do
  nxt=$(( cur + step_sec ))
  if (( nxt > end_epoch )); then
    nxt="${end_epoch}"
  fi
  s="$(to_isoz "${cur}")"
  e="$(to_isoz "${nxt}")"
  echo "[merge_chunk] idx=${idx} start=${s} end=${e}"
  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/merge_feature_views.py"
    --database-url "${DATABASE_URL}"
    --start "${s}"
    --end "${e}"
  )
  if [[ "${DO_TRUNCATE}" == "1" && "${first}" == "1" ]]; then
    cmd+=(--truncate)
  fi
  "${cmd[@]}"
  first=0
  idx=$(( idx + 1 ))
  cur="${nxt}"
done

echo "[merge_chunk] done chunks=${idx}"
