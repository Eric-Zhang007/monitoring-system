#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUNDLE_DIR=""
SERVICES="backend collector model_ops task_worker nginx redis postgres prometheus alertmanager grafana frontend"
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --services)
      SERVICES="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUNDLE_DIR" ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  BUNDLE_DIR="$ROOT_DIR/artifacts/server_bundle/$ts"
fi

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

mkdir -p "$BUNDLE_DIR"
services_arr=($SERVICES)

if [[ "$SKIP_BUILD" != "1" ]]; then
  compose_cmd build "${services_arr[@]}"
fi

mapfile -t images < <(compose_cmd config --images | sort -u)
if [[ ${#images[@]} -eq 0 ]]; then
  echo "no images resolved from compose" >&2
  exit 2
fi

docker save -o "$BUNDLE_DIR/docker-images.tar" "${images[@]}"

mkdir -p "$BUNDLE_DIR/nginx" "$BUNDLE_DIR/monitoring" "$BUNDLE_DIR/scripts"
cp docker-compose.yml "$BUNDLE_DIR/docker-compose.yml"
cp nginx/nginx.conf "$BUNDLE_DIR/nginx/nginx.conf"
cp monitoring/prometheus.yml "$BUNDLE_DIR/monitoring/prometheus.yml"
cp monitoring/alerts.yml "$BUNDLE_DIR/monitoring/alerts.yml"
cp monitoring/alertmanager.yml "$BUNDLE_DIR/monitoring/alertmanager.yml"
cp scripts/server_seed_db.sh "$BUNDLE_DIR/scripts/server_seed_db.sh"
cp scripts/server_bootstrap.sh "$BUNDLE_DIR/scripts/server_bootstrap.sh"
cp scripts/server_verify_runtime.sh "$BUNDLE_DIR/scripts/server_verify_runtime.sh"
cp scripts/init_db.sql "$BUNDLE_DIR/scripts/init_db.sql"

python3 - <<PY
import json, subprocess, pathlib, datetime
bundle = pathlib.Path(r"$BUNDLE_DIR")
images = r"""${images[*]}""".split()
rev = "unknown"
try:
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    pass
manifest = {
    "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "git_revision": rev,
    "images": images,
    "files": sorted([str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()]),
}
(bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

( cd "$(dirname "$BUNDLE_DIR")" && tar -czf "$(basename "$BUNDLE_DIR").tar.gz" "$(basename "$BUNDLE_DIR")" )

echo "[OK] bundle created"
echo "bundle_dir=$BUNDLE_DIR"
echo "bundle_tar=${BUNDLE_DIR}.tar.gz"
