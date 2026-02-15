#!/usr/bin/env python3
"""
System Monitoring Script
Monitors GPU, memory, and service health
"""
import time
import subprocess
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_service_health(service_name: str) -> bool:
    """Check if a Docker service is healthy"""
    try:
        result = subprocess.run(
            ['docker', 'inspect', '--format={{.State.Health.Status}}', service_name],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == 'healthy'
    except Exception as e:
        logger.error(f"Failed to check {service_name}: {e}")
        return False

def get_gpu_status() -> dict:
    """Get GPU status using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            gpu_data = []
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                gpu_data.append({
                    'index': parts[0],
                    'name': parts[1],
                    'memory_used_mb': parts[2],
                    'memory_total_mb': parts[3],
                    'utilization_percent': parts[4]
                })
            return {'gpus': gpu_data, 'status': 'ok'}
        return {'gpus': [], 'status': 'not_available'}
    except Exception as e:
        logger.warning(f"GPU monitoring failed: {e}")
        return {'gpus': [], 'status': 'error'}

def get_system_memory() -> dict:
    """Get system memory usage"""
    try:
        result = subprocess.run(
            ['free', '-m'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith('Mem:'):
                parts = line.split()
                total = int(parts[1])
                used = int(parts[2])
                percent = int(parts[2]) / int(parts[1]) * 100
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'percent': round(percent, 1),
                    'status': 'ok'
                }
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
    return {'status': 'error'}

def monitor_loop(interval: int = 30):
    """Main monitoring loop"""
    services = ['monitoring-system-backend-1', 'monitoring_system-frontend-1', 'monitoring-system-inference-1']
    
    logger.info("🔍 Starting system monitoring...")
    logger.info(f"Watching services: {', '.join(services)}")
    
    while True:
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check services
            for service in services:
                is_healthy = check_service_health(service)
                status = '✅ healthy' if is_healthy else '❌ unhealthy'
                logger.info(f"[{timestamp}] {service}: {status}")
            
            # Check GPU
            gpu_status = get_gpu_status()
            if gpu_status['status'] == 'ok':
                for gpu in gpu_status['gpus']:
                    logger.info(f"[{timestamp}] GPU {gpu['index']} ({gpu['name']}): {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB ({gpu['utilization_percent']}% util)")
            else:
                logger.warning(f"[{timestamp}] GPU status: {gpu_status['status']}")
            
            # Check memory
            mem_status = get_system_memory()
            if mem_status['status'] == 'ok':
                logger.info(f"[{timestamp}] Memory: {mem_status['used_mb']}MB / {mem_status['total_mb']}MB ({mem_status['percent']}%)")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("❌ Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
            time.sleep(interval)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monitor monitoring system services')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    args = parser.parse_args()
    monitor_loop(interval=args.interval)
