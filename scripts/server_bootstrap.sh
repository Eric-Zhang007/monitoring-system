#!/usr/bin/env bash
set -euo pipefail

BUNDLE_TAR=""
DEPLOY_DIR="/opt/monitoring-system/current"
ENV_FILE=""
DB_DUMP=""
SKIP_IMAGE_LOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      BUNDLE_TAR="$2"
      shift 2
      ;;
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --db-dump)
      DB_DUMP="$2"
      shift 2
      ;;
    --skip-image-load)
      SKIP_IMAGE_LOAD=1
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUNDLE_TAR" || ! -f "$BUNDLE_TAR" ]]; then
  echo "usage: $0 --bundle <bundle.tar.gz> [--deploy-dir /opt/monitoring-system/current] [--env-file .env] [--db-dump dump.fc]" >&2
  exit 2
fi

mkdir -p "$DEPLOY_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

tar -xzf "$BUNDLE_TAR" -C "$TMP_DIR"
BUNDLE_ROOT="$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)"
if [[ -z "$BUNDLE_ROOT" ]]; then
  echo "invalid bundle: cannot find root dir" >&2
  exit 2
fi

cp "$BUNDLE_ROOT/docker-compose.yml" "$DEPLOY_DIR/docker-compose.yml"
mkdir -p "$DEPLOY_DIR/nginx" "$DEPLOY_DIR/monitoring" "$DEPLOY_DIR/scripts"
cp "$BUNDLE_ROOT/nginx/nginx.conf" "$DEPLOY_DIR/nginx/nginx.conf"
cp "$BUNDLE_ROOT/monitoring/"*.yml "$DEPLOY_DIR/monitoring/"
cp "$BUNDLE_ROOT/scripts/"*.sh "$DEPLOY_DIR/scripts/"
cp "$BUNDLE_ROOT/scripts/init_db.sql" "$DEPLOY_DIR/scripts/init_db.sql"
chmod +x "$DEPLOY_DIR/scripts/"*.sh

if [[ -n "$ENV_FILE" ]]; then
  cp "$ENV_FILE" "$DEPLOY_DIR/.env"
fi

if [[ "$SKIP_IMAGE_LOAD" != "1" ]]; then
  docker load -i "$BUNDLE_ROOT/docker-images.tar"
fi

compose=(docker compose -f "$DEPLOY_DIR/docker-compose.yml")
if [[ -f "$DEPLOY_DIR/.env" ]]; then
  compose=(docker compose --env-file "$DEPLOY_DIR/.env" -f "$DEPLOY_DIR/docker-compose.yml")
fi

"${compose[@]}" up -d --remove-orphans
"${compose[@]}" exec -T backend bash -lc 'cd /app && alembic upgrade head'

if [[ -n "$DB_DUMP" ]]; then
  if [[ ! -f "$DB_DUMP" ]]; then
    echo "db dump not found: $DB_DUMP" >&2
    exit 2
  fi
  cat "$DB_DUMP" | "${compose[@]}" exec -T postgres pg_restore -U monitor -d monitor --clean --if-exists --no-owner --no-privileges
fi

echo "[OK] bootstrap done"
echo "deploy_dir=$DEPLOY_DIR"
