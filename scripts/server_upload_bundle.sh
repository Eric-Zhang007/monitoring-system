#!/usr/bin/env bash
set -euo pipefail

BUNDLE_TAR=""
HOST=""
USER_NAME="root"
PORT="22"
REMOTE_DIR="/opt/monitoring-system/bundles"
DB_DUMP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      BUNDLE_TAR="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --db-dump)
      DB_DUMP="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUNDLE_TAR" || -z "$HOST" ]]; then
  echo "usage: $0 --bundle <bundle.tar.gz> --host <host> [--user root] [--port 22] [--remote-dir /opt/monitoring-system/bundles] [--db-dump path]" >&2
  exit 2
fi
if [[ ! -f "$BUNDLE_TAR" ]]; then
  echo "bundle not found: $BUNDLE_TAR" >&2
  exit 2
fi

ssh -p "$PORT" "$USER_NAME@$HOST" "mkdir -p '$REMOTE_DIR'"
scp -P "$PORT" "$BUNDLE_TAR" "$USER_NAME@$HOST:$REMOTE_DIR/"

if [[ -n "$DB_DUMP" ]]; then
  if [[ ! -f "$DB_DUMP" ]]; then
    echo "db dump not found: $DB_DUMP" >&2
    exit 2
  fi
  scp -P "$PORT" "$DB_DUMP" "$USER_NAME@$HOST:$REMOTE_DIR/"
fi

echo "[OK] uploaded"
echo "remote_bundle=$REMOTE_DIR/$(basename "$BUNDLE_TAR")"
if [[ -n "$DB_DUMP" ]]; then
  echo "remote_db_dump=$REMOTE_DIR/$(basename "$DB_DUMP")"
fi
