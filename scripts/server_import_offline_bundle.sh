#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"

HOST="${HOST:-}"
PORT="${PORT:-22}"
USER_NAME="${USER_NAME:-root}"
REMOTE_DIR="${REMOTE_DIR:-/opt/monitoring-system}"
DB_URL="${DB_URL:-postgresql://monitor@localhost:5432/monitor}"
PRIMARY_TF="${PRIMARY_TF:-5m}"
SYMBOLS="${SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
BUNDLE_TAR="${BUNDLE_TAR:-}"
REMOTE_BUNDLE_DIR="${REMOTE_BUNDLE_DIR:-}"
PYTHON_BIN_REMOTE="${PYTHON_BIN_REMOTE:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --db-url) DB_URL="$2"; shift 2 ;;
    --primary-timeframe) PRIMARY_TF="$2"; shift 2 ;;
    --symbols) SYMBOLS="$2"; shift 2 ;;
    --bundle-tar) BUNDLE_TAR="$2"; shift 2 ;;
    --remote-bundle-dir) REMOTE_BUNDLE_DIR="$2"; shift 2 ;;
    --python-bin-remote) PYTHON_BIN_REMOTE="$2"; shift 2 ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "[ERR] --host is required" >&2
  exit 2
fi
if [[ -z "$BUNDLE_TAR" && -z "$REMOTE_BUNDLE_DIR" ]]; then
  echo "[ERR] provide --bundle-tar (upload) or --remote-bundle-dir (already on server)" >&2
  exit 2
fi
if [[ -n "$BUNDLE_TAR" && ! -f "$BUNDLE_TAR" ]]; then
  echo "[ERR] bundle tar not found: $BUNDLE_TAR" >&2
  exit 2
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER_NAME@$HOST")
SCP_BASE=(scp -o StrictHostKeyChecking=no -P "$PORT")
if [[ -n "${SSHPASS:-}" ]]; then
  SSH_BASE=(sshpass -p "$SSHPASS" "${SSH_BASE[@]}")
  SCP_BASE=(sshpass -p "$SSHPASS" "${SCP_BASE[@]}")
fi

if [[ -n "$BUNDLE_TAR" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  REMOTE_IMPORT_DIR="${REMOTE_DIR%/}/artifacts/offline_bundle/import_${RUN_ID}"
  REMOTE_TAR="${REMOTE_IMPORT_DIR}/$(basename "$BUNDLE_TAR")"

  echo "[1/4] prepare remote dir"
  "${SSH_BASE[@]}" "mkdir -p '$REMOTE_IMPORT_DIR'"

  echo "[2/4] upload bundle tar"
  "${SCP_BASE[@]}" "$BUNDLE_TAR" "$USER_NAME@$HOST:$REMOTE_TAR"

  echo "[3/4] extract bundle"
  "${SSH_BASE[@]}" "cd '$REMOTE_IMPORT_DIR' && tar -xzf '$REMOTE_TAR'"

  REMOTE_BUNDLE_DIR="$REMOTE_IMPORT_DIR"
else
  echo "[1/1] use existing remote bundle dir: $REMOTE_BUNDLE_DIR"
fi

echo "[4/4] run import_offline_data_bundle.sh on server"
"${SSH_BASE[@]}" "cd '$REMOTE_DIR' && PYTHON_BIN='$PYTHON_BIN_REMOTE' DATABASE_URL='$DB_URL' PRIMARY_TF='$PRIMARY_TF' SYMBOLS='$SYMBOLS' bash scripts/import_offline_data_bundle.sh --bundle-dir '$REMOTE_BUNDLE_DIR' --database-url '$DB_URL' --primary-timeframe '$PRIMARY_TF' --symbols '$SYMBOLS'"

echo "done"
echo "remote_bundle_dir=$REMOTE_BUNDLE_DIR"
