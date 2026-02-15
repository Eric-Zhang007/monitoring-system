"""
Health Check Script - 系统健康监控
"""
import os
import sys
import asyncio
import aiohttp
import psycopg2
import redis
import requests
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
SERVICES = {
    'backend': 'http://localhost:8000/health',
    'redis': 'redis://localhost:6379',
    'postgres': 'postgresql://monitor:change_me_please@localhost:5432/monitor',
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
