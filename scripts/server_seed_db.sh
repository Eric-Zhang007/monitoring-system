#!/usr/bin/env bash
set -euo pipefail

MODE=""
IN_FILE=""
OUT_FILE=""
DB_SERVICE="postgres"
DB_USER="monitor"
DB_NAME="monitor"

while [[ $# -gt 0 ]]; do
  case "$1" in
    export|import)
      MODE="$1"
      shift
      ;;
    --in)
      IN_FILE="$2"
      shift 2
      ;;
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    --db-service)
      DB_SERVICE="$2"
      shift 2
      ;;
    --db-user)
      DB_USER="$2"
      shift 2
      ;;
    --db-name)
      DB_NAME="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi
  echo "docker compose not found" >&2
  return 127
}

if [[ "$MODE" == "export" ]]; then
  if [[ -z "$OUT_FILE" ]]; then
    echo "export requires --out <file>" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$OUT_FILE")"
  compose_cmd exec -T "$DB_SERVICE" pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc > "$OUT_FILE"
  echo "[OK] exported db dump: $OUT_FILE"
  exit 0
fi

if [[ "$MODE" == "import" ]]; then
  if [[ -z "$IN_FILE" || ! -f "$IN_FILE" ]]; then
    echo "import requires --in <existing file>" >&2
    exit 2
  fi
  cat "$IN_FILE" | compose_cmd exec -T "$DB_SERVICE" pg_restore -U "$DB_USER" -d "$DB_NAME" --clean --if-exists --no-owner --no-privileges
  echo "[OK] imported db dump: $IN_FILE"
  exit 0
fi

echo "usage: $0 export --out dump.fc | import --in dump.fc" >&2
exit 2
