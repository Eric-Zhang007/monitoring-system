"""
Health Check Script - 系统健康监控
"""
import os
import sys
import asyncio
import aiohttp
import psycopg2
import redis
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
SERVICES = {
    'backend': 'http://localhost:8000/health',
    'metrics': 'http://localhost:8000/metrics',
    'collector_metrics': os.getenv('COLLECTOR_METRICS_URL', 'http://localhost:9101/metrics'),
    'redis': 'redis://localhost:6379',
    'postgres': 'postgresql://monitor:change_me_please@localhost:5432/monitor',
    'prometheus': 'http://localhost:9090/-/healthy',
    'prometheus_query': os.getenv('PROMETHEUS_QUERY_URL', 'http://localhost:9090/api/v1/query'),
    'grafana': 'http://localhost:3000/api/health',
}

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def log_success(service: str, message: str):
    print(f"{Colors.GREEN}✅ {service}: {message}{Colors.RESET}")


def log_error(service: str, message: str):
    print(f"{Colors.RED}❌ {service}: {message}{Colors.RESET}")


def log_warning(service: str, message: str):
    print(f"{Colors.YELLOW}⚠️ {service}: {message}{Colors.RESET}")


def log_info(message: str):
    print(f"{Colors.BOLD}{message}{Colors.RESET}")


async def check_backend() -> Tuple[bool, Dict]:
    """Check backend service health"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SERVICES['backend'], timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return True, data
                return False, {"error": f"HTTP {response.status}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_redis() -> Tuple[bool, Dict]:
    """Check Redis connection"""
    try:
        r = redis.from_url(SERVICES['redis'])
        r.ping()
        info = r.info()
        return True, {
            "connected": True,
            "version": info.get("redis_version"),
            "memory": info.get("used_memory_human"),
        }
    except Exception as e:
        return False, {"error": str(e)}


def check_postgres() -> Tuple[bool, Dict]:
    """Check PostgreSQL connection"""
    try:
        conn = psycopg2.connect(SERVICES['postgres'])
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        # Check recent records
        cursor.execute("SELECT COUNT(*) FROM predictions_v2 WHERE created_at > NOW() - make_interval(hours => 1);")
        recent_predictions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM semantic_features WHERE timestamp > NOW() - make_interval(hours => 6);")
        recent_features = cursor.fetchone()[0]
        
        conn.close()
        return True, {
            "version": version[0],
            "recent_predictions": recent_predictions,
            "recent_features": recent_features,
        }
    except Exception as e:
        return False, {"error": str(e)}


async def check_grafana() -> Tuple[bool, Dict]:
    """Check Grafana health"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SERVICES['grafana'], timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return True, data
                return False, {"error": f"HTTP {response.status}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_prometheus() -> Tuple[bool, Dict]:
    """Check Prometheus health"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SERVICES['prometheus'], timeout=10) as response:
                if response.status == 200:
                    return True, {"status": "healthy"}
                return False, {"error": f"HTTP {response.status}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_metrics() -> Tuple[bool, Dict]:
    """Check backend Prometheus metrics endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SERVICES['metrics'], timeout=10) as response:
                if response.status != 200:
                    return False, {"error": f"HTTP {response.status}"}
                txt = await response.text()
                required = [
                    "ms_http_requests_total",
                    "ms_execution_orders_total",
                    "ms_signal_latency_seconds_bucket",
                    "ms_execution_latency_seconds_bucket",
                    "ms_execution_reject_rate",
                    "ms_data_freshness_seconds",
                    "ms_model_drift_events_total",
                    "ms_risk_hard_blocks_total",
                ]
                found = all(k in txt for k in required)
                if not found:
                    return False, {"error": "required metrics not found"}
                slo = evaluate_slo_from_metrics(txt)
                return True, {"status": "ok", "slo": slo}
    except Exception as e:
        return False, {"error": str(e)}


def _counter_by_label(metrics_text: str, metric_name: str, label_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line.startswith(f"{metric_name}" + "{"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        left, right = parts
        token = f'{label_name}="'
        if token not in left:
            continue
        label_val = left.split(token, 1)[1].split('"', 1)[0]
        try:
            value = float(right)
        except ValueError:
            continue
        out[label_val] = out.get(label_val, 0.0) + value
    return out


def _parse_histogram_buckets(metrics_text: str, metric_prefix: str) -> Tuple[List[Tuple[float, float]], float]:
    bucket_map: Dict[float, float] = {}
    total_count = 0.0
    for line in metrics_text.splitlines():
        if not line.startswith(f"{metric_prefix}_bucket"):
            continue
        left = line.split(" ", 1)[0]
        val = float(line.rsplit(" ", 1)[-1]) if " " in line else 0.0
        if 'le="' not in left:
            continue
        le_raw = left.split('le="', 1)[1].split('"', 1)[0]
        if le_raw == "+Inf":
            total_count += val
            continue
        try:
            le = float(le_raw)
        except ValueError:
            continue
        bucket_map[le] = bucket_map.get(le, 0.0) + val
    buckets = sorted(bucket_map.items(), key=lambda x: x[0])
    if total_count <= 0 and buckets:
        total_count = buckets[-1][1]
    return buckets, total_count


def _histogram_quantile_approx(buckets: List[Tuple[float, float]], q: float) -> float:
    if not buckets:
        return 0.0
    total = buckets[-1][1]
    if total <= 0:
        return 0.0
    target = total * q
    for le, cumulative in buckets:
        if cumulative >= target:
            return le
    return buckets[-1][0]


def _http_availability(metrics_text: str) -> float:
    total = 0.0
    errors = 0.0
    for line in metrics_text.splitlines():
        if not line.startswith("ms_http_requests_total{"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        left, right = parts
        try:
            count = float(right)
        except ValueError:
            continue
        total += count
        if 'status="' in left:
            status = left.split('status="', 1)[1].split('"', 1)[0]
            if status.startswith("5"):
                errors += count
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, (total - errors) / total))


def evaluate_slo_from_metrics(metrics_text: str) -> Dict[str, Dict[str, object]]:
    signal_buckets, signal_count = _parse_histogram_buckets(metrics_text, "ms_signal_latency_seconds")
    exec_buckets, exec_count = _parse_histogram_buckets(metrics_text, "ms_execution_latency_seconds")
    signal_state: Dict[str, object]
    exec_state: Dict[str, object]
    availability = _http_availability(metrics_text)
    if signal_count <= 0:
        signal_state = {
            "status": "insufficient_observation",
            "p50_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "target_p95_seconds": 0.150,
        }
    else:
        p50 = _histogram_quantile_approx(signal_buckets, 0.50)
        p95 = _histogram_quantile_approx(signal_buckets, 0.95)
        p99 = _histogram_quantile_approx(signal_buckets, 0.99)
        signal_state = {
            "status": "pass" if p95 < 0.150 else "degraded",
            "p50_seconds": p50,
            "p95_seconds": p95,
            "p99_seconds": p99,
            "target_p95_seconds": 0.150,
        }
    if exec_count <= 0:
        exec_state = {
            "status": "insufficient_observation",
            "p50_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "target_p95_seconds": 0.300,
        }
    else:
        p50 = _histogram_quantile_approx(exec_buckets, 0.50)
        p95 = _histogram_quantile_approx(exec_buckets, 0.95)
        p99 = _histogram_quantile_approx(exec_buckets, 0.99)
        exec_state = {
            "status": "pass" if p95 < 0.300 else "degraded",
            "p50_seconds": p50,
            "p95_seconds": p95,
            "p99_seconds": p99,
            "target_p95_seconds": 0.300,
        }
    overall = "pass"
    if signal_state["status"] == "degraded" or exec_state["status"] == "degraded":
        overall = "degraded"
    elif signal_state["status"] == "insufficient_observation" or exec_state["status"] == "insufficient_observation":
        overall = "insufficient_observation"
    availability_state = {
        "status": "pass" if availability >= 0.999 else "degraded",
        "value": round(availability, 6),
        "availability_5m": round(availability, 6),
        "availability_1h": round(availability, 6),
        "target": 0.999,
    }
    blocking_reason = "none"
    if availability_state["status"] == "degraded" and overall == "pass":
        overall = "degraded"
    if signal_state["status"] == "degraded":
        blocking_reason = "signal_latency_p95"
    elif exec_state["status"] == "degraded":
        blocking_reason = "execution_latency_p95"
    elif availability_state["status"] == "degraded":
        blocking_reason = "api_availability"
    elif signal_state["status"] == "insufficient_observation" or exec_state["status"] == "insufficient_observation":
        blocking_reason = "insufficient_observation"
    return {
        "overall": {"status": overall, "slo_blocking_reason": blocking_reason},
        "signal_latency": signal_state,
        "execution_latency": exec_state,
        "api_availability": availability_state,
    }


def evaluate_collector_slo_from_metrics(metrics_text: str) -> Dict[str, Dict[str, object]]:
    fetch_by_status = _counter_by_label(metrics_text, "ms_collector_connector_fetch_total", "status")
    success = float(fetch_by_status.get("success", 0.0))
    empty = float(fetch_by_status.get("empty", 0.0))
    failure = float(fetch_by_status.get("failure", 0.0))
    attempts = success + empty + failure
    if attempts <= 0:
        connector_state: Dict[str, object] = {
            "status": "insufficient_observation",
            "success_rate": None,
            "target_success_rate": 0.95,
            "attempts": 0,
        }
    else:
        success_rate = (success + empty) / attempts
        connector_state = {
            "status": "pass" if success_rate >= 0.95 else "degraded",
            "success_rate": round(success_rate, 6),
            "target_success_rate": 0.95,
            "attempts": int(attempts),
        }

    publish_buckets, publish_count = _parse_histogram_buckets(
        metrics_text, "ms_collector_source_publish_to_ingest_seconds"
    )
    if publish_count <= 0:
        latency_state: Dict[str, object] = {
            "status": "insufficient_observation",
            "p95_seconds": None,
            "target_p95_seconds": 120.0,
        }
    else:
        p95 = _histogram_quantile_approx(publish_buckets, 0.95)
        latency_state = {
            "status": "pass" if p95 < 120.0 else "degraded",
            "p95_seconds": p95,
            "target_p95_seconds": 120.0,
        }

    overall = "pass"
    if connector_state["status"] == "degraded" or latency_state["status"] == "degraded":
        overall = "degraded"
    elif connector_state["status"] == "insufficient_observation" or latency_state["status"] == "insufficient_observation":
        overall = "insufficient_observation"

    blocking_reason = "none"
    if connector_state["status"] == "degraded":
        blocking_reason = "connector_success_rate"
    elif latency_state["status"] == "degraded":
        blocking_reason = "source_publish_to_ingest_p95"
    elif connector_state["status"] == "insufficient_observation" or latency_state["status"] == "insufficient_observation":
        blocking_reason = "insufficient_observation"

    return {
        "overall": {"status": overall, "slo_blocking_reason": blocking_reason},
        "connector_success_rate": connector_state,
        "source_publish_to_ingest": latency_state,
    }


async def check_collector_metrics() -> Tuple[bool, Dict]:
    """Check collector Prometheus metrics endpoint with Prometheus fallback."""
    required = [
        "ms_collector_connector_fetch_total",
        "ms_collector_connector_rate_limit_total",
        "ms_collector_event_publish_total",
    ]
    endpoint_error = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SERVICES['collector_metrics'], timeout=8) as response:
                if response.status == 200:
                    txt = await response.text()
                    missing = [k for k in required if k not in txt]
                    if missing:
                        return False, {"error": f"collector metrics missing keys: {','.join(missing)}"}
                    slo = evaluate_collector_slo_from_metrics(txt)
                    return True, {"status": "ok", "source": "collector_endpoint", "slo": slo}
                endpoint_error = f"collector endpoint HTTP {response.status}"
    except Exception as e:
        endpoint_error = str(e)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SERVICES["prometheus_query"],
                params={"query": 'up{job="collector"}'},
                timeout=8,
            ) as response:
                if response.status != 200:
                    return False, {"error": f"{endpoint_error}; prometheus query HTTP {response.status}"}
                data = await response.json()
                result = (((data or {}).get("data") or {}).get("result") or [])
                is_up = False
                for item in result:
                    value = item.get("value") if isinstance(item, dict) else None
                    if isinstance(value, list) and len(value) >= 2 and str(value[1]) == "1":
                        is_up = True
                        break
                if is_up:
                    return True, {
                        "status": "ok",
                        "source": "prometheus_up_fallback",
                        "warning": f"collector endpoint unreachable: {endpoint_error}",
                    }
                return False, {"error": f"{endpoint_error}; collector up metric not healthy"}
    except Exception as e:
        return False, {"error": f"{endpoint_error}; prometheus query failed: {e}"}


def check_docker_services():
    """Check Docker container status"""
    import subprocess
    services = [
        'monitoring-system-backend-1',
        'monitoring-system-inference-1',
        'monitoring-system-training-1',
        'monitoring-system-collector-1',
        'monitoring-system-redis-1',
        'monitoring-system-postgres-1',
    ]
    
    log_info("\n🐳 Docker Container Status")
    print("=" * 50)
    
    for service in services:
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={service}', '--format', '{{.Status}}'],
                capture_output=True, text=True, timeout=5
            )
            status = result.stdout.strip()
            if 'Up' in status:
                log_success(service.split('-')[-2], f"Running - {status}")
            else:
                log_error(service.split('-')[-2], "Not running")
        except Exception as e:
            log_error(service.split('-')[-2], f"Error: {e}")


async def run_health_checks():
    """Run all health checks"""
    log_info("\n🔍 System Health Check")
    print("=" * 50)
    print(f"Time: {datetime.now().isoformat()}\n")
    
    all_healthy = True
    
    # Check Backend
    log_info("\n📡 Checking Backend Service...")
    healthy, data = await check_backend()
    if healthy:
        log_success('backend', f"Healthy - GPU: {data.get('gpu', {}).get('available_gpus', 'N/A')}")
    else:
        log_error('backend', data.get('error', 'Unknown error'))
        all_healthy = False
    
    # Check Redis
    log_info("\n📦 Checking Redis...")
    healthy, data = await check_redis()
    if healthy:
        log_success('redis', f"Connected - Version {data.get('version')}, Memory: {data.get('memory')}")
    else:
        log_error('redis', data.get('error', 'Connection failed'))
        all_healthy = False
    
    # Check PostgreSQL
    log_info("\n🗄️  Checking PostgreSQL...")
    healthy, data = await check_postgres()
    if healthy:
        log_success('postgres', f"Connected - Recent predictions: {data.get('recent_predictions')}, Features: {data.get('recent_features')}")
    else:
        log_error('postgres', data.get('error', 'Connection failed'))
        all_healthy = False
    
    # Check Grafana
    log_info("\n📊 Checking Grafana...")
    healthy, data = await check_grafana()
    if healthy:
        log_success('grafana', "Healthy")
    else:
        log_warning('grafana', data.get('error', 'Connection failed (may not be critical)'))

    # Check Prometheus
    log_info("\n📉 Checking Prometheus...")
    healthy, data = await check_prometheus()
    if healthy:
        log_success('prometheus', "Healthy")
    else:
        log_warning('prometheus', data.get('error', 'Connection failed (may not be critical)'))

    # Check Metrics
    log_info("\n📈 Checking Prometheus Metrics...")
    healthy, data = await check_metrics()
    if healthy:
        log_success('metrics', "Endpoint healthy")
        slo = data.get("slo", {})
        s = slo.get("signal_latency", {})
        e = slo.get("execution_latency", {})
        a = slo.get("api_availability", {})
        print(f"  SLO(signal p95<150ms): status={s.get('status')} p50={s.get('p50_seconds')} p95={s.get('p95_seconds')} p99={s.get('p99_seconds')}")
        print(f"  SLO(execution p95<300ms): status={e.get('status')} p50={e.get('p50_seconds')} p95={e.get('p95_seconds')} p99={e.get('p99_seconds')}")
        print(
            f"  SLO(api availability>=99.9%): status={a.get('status')} "
            f"value={a.get('value')} 5m={a.get('availability_5m')} 1h={a.get('availability_1h')}"
        )
        print(
            f"  SLO(overall): {slo.get('overall', {}).get('status')} "
            f"reason={slo.get('overall', {}).get('slo_blocking_reason')}"
        )
    else:
        log_warning('metrics', data.get('error', 'Metrics endpoint unhealthy'))

    # Check Collector Metrics
    log_info("\n🛰️  Checking Collector Metrics...")
    healthy, data = await check_collector_metrics()
    if healthy:
        log_success('collector_metrics', f"Healthy ({data.get('source', 'unknown')})")
        if data.get("warning"):
            print(f"  warning: {data.get('warning')}")
        collector_slo = data.get("slo", {})
        if collector_slo:
            c = collector_slo.get("connector_success_rate", {})
            p = collector_slo.get("source_publish_to_ingest", {})
            print(
                f"  SLO(connector success>=95%): status={c.get('status')} "
                f"rate={c.get('success_rate')} attempts={c.get('attempts')}"
            )
            print(
                f"  SLO(source->ingest p95<120s): status={p.get('status')} "
                f"p95={p.get('p95_seconds')}"
            )
            print(
                f"  SLO(collector overall): {collector_slo.get('overall', {}).get('status')} "
                f"reason={collector_slo.get('overall', {}).get('slo_blocking_reason')}"
            )
    else:
        log_warning('collector_metrics', data.get('error', 'Collector metrics unhealthy'))
    
    # Check Docker
    check_docker_services()
    
    print("\n" + "=" * 50)
    if all_healthy:
        log_info(f"{Colors.GREEN}✅ All critical services are healthy!{Colors.RESET}")
        return 0
    else:
        log_error("Health Check", "Some services are unhealthy. Please investigate.")
        return 1


if __name__ == "__main__":
    result = asyncio.run(run_health_checks())
    sys.exit(result)
