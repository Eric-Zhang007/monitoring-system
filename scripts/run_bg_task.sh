#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ROOT_DIR}/artifacts/runtime/bg"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_bg_task.sh start <name> -- <command ...>
  scripts/run_bg_task.sh status <name>
  scripts/run_bg_task.sh stop <name>
  scripts/run_bg_task.sh tail <name> [lines]
EOF
}

require_name() {
  local n="${1:-}"
  if [[ -z "${n}" ]]; then
    echo "task name is required"
    usage
    exit 2
  fi
}

is_alive() {
  local pid="${1:-0}"
  kill -0 "${pid}" >/dev/null 2>&1
}

meta_value() {
  local file="${1:-}"
  local key="${2:-}"
  if [[ ! -f "${file}" ]] || [[ -z "${key}" ]]; then
    return 0
  fi
  awk -F= -v k="${key}" '$1==k {sub($1"=",""); print; exit}' "${file}"
}

cmd="${1:-}"
case "${cmd}" in
  start)
    shift || true
    name="${1:-}"
    require_name "${name}"
    shift || true
    if [[ "${1:-}" != "--" ]]; then
      echo "missing '--' before command"
      usage
      exit 2
    fi
    shift || true
    if [[ "$#" -eq 0 ]]; then
      echo "missing command"
      usage
      exit 2
    fi
    pid_file="${PID_DIR}/${name}.pid"
    if [[ -f "${pid_file}" ]]; then
      old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [[ -n "${old_pid}" ]] && is_alive "${old_pid}"; then
        echo "task_already_running name=${name} pid=${old_pid}"
        exit 1
      fi
      rm -f "${pid_file}"
    fi
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    log_file="${LOG_DIR}/${name}_${ts}.log"
    nohup "$@" >"${log_file}" 2>&1 &
    pid="$!"
    cmd_escaped="$(printf '%q ' "$@")"
    echo "${pid}" > "${pid_file}"
    meta_file="${PID_DIR}/${name}.meta"
    {
      echo "PID=${pid}"
      echo "LOG_FILE=${log_file}"
      echo "STARTED_AT=${ts}"
      echo "CMD=${cmd_escaped}"
    } > "${meta_file}"
    echo "started name=${name} pid=${pid} log=${log_file}"
    ;;

  status)
    shift || true
    name="${1:-}"
    require_name "${name}"
    pid_file="${PID_DIR}/${name}.pid"
    meta_file="${PID_DIR}/${name}.meta"
    if [[ ! -f "${pid_file}" ]]; then
      echo "not_running name=${name}"
      log_file="$(meta_value "${meta_file}" "LOG_FILE" || true)"
      if [[ -n "${log_file}" ]]; then
        echo "last_log=${log_file}"
      fi
      exit 0
    fi
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -z "${pid}" ]]; then
      echo "invalid_pid_file name=${name}"
      exit 1
    fi
    log_file="$(meta_value "${meta_file}" "LOG_FILE" || true)"
    if is_alive "${pid}"; then
      echo "running name=${name} pid=${pid}"
      if [[ -n "${log_file}" ]]; then
        echo "log=${log_file}"
        tail -n 5 "${log_file}" 2>/dev/null || true
      fi
    else
      echo "stopped name=${name} pid=${pid}"
      if [[ -n "${log_file}" ]]; then
        echo "last_log=${log_file}"
        tail -n 10 "${log_file}" 2>/dev/null || true
      fi
      exit 1
    fi
    ;;

  stop)
    shift || true
    name="${1:-}"
    require_name "${name}"
    pid_file="${PID_DIR}/${name}.pid"
    if [[ ! -f "${pid_file}" ]]; then
      echo "not_running name=${name}"
      exit 0
    fi
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -z "${pid}" ]]; then
      rm -f "${pid_file}"
      echo "invalid_pid_file name=${name}"
      exit 1
    fi
    if is_alive "${pid}"; then
      kill "${pid}"
      for _ in $(seq 1 20); do
        if ! is_alive "${pid}"; then
          break
        fi
        sleep 0.25
      done
      if is_alive "${pid}"; then
        echo "failed_to_stop name=${name} pid=${pid}"
        exit 1
      fi
    fi
    rm -f "${pid_file}"
    echo "stopped name=${name} pid=${pid}"
    ;;

  tail)
    shift || true
    name="${1:-}"
    require_name "${name}"
    lines="${2:-80}"
    meta_file="${PID_DIR}/${name}.meta"
    log_file="$(meta_value "${meta_file}" "LOG_FILE" || true)"
    if [[ -z "${log_file}" ]]; then
      echo "missing_log_metadata name=${name}"
      exit 1
    fi
    if [[ ! -f "${log_file}" ]]; then
      echo "missing_log_file name=${name} path=${log_file}"
      exit 1
    fi
    tail -n "${lines}" "${log_file}"
    ;;

  *)
    usage
    exit 2
    ;;
esac
