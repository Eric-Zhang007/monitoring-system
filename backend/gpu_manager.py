"""
GPU Resource Manager (v3 - Fixed)
- GPU 0: Real-time inference
- GPU 1: Training + Offline tasks (NIM feature extraction)
- Includes system memory monitoring
"""
import torch
import pynvml
import psutil
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class GPUManager:
    """Dual A100 Manager (Fixed v3)"""

    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(0),  # Real-time inference
                pynvml.nvmlDeviceGetHandleByIndex(1),  # Training + Offline
            ]
        except Exception as e:
            logger.warning(f"NVML not available: {e}")
            self.gpu_handles = []
            logger.info("Running in CPU-only mode")

        # Memory thresholds (GB)
        self.memory_thresholds = {
            0: {"warning": 20, "critical": 25},  # GPU 0: Real-time, conservative
            1: {"warning": 35, "critical": 42},  # GPU 1: Training
        }

    def get_device(self, task_type: Literal["inference", "training"]) -> int:
        """Allocate GPU based on task type"""
        if task_type == "inference":
            return 0
        else:
            return 1

    def check_memory(self, device_id: int) -> dict:
        """Check GPU memory usage"""
        if device_id >= len(self.gpu_handles):
            logger.warning(f"GPU {device_id} not available")
            return {
                "device": device_id,
                "used_gb": 0,
                "total_gb": 0,
                "usage_percent": 0,
                "status": "unavailable"
            }

        handle = self.gpu_handles[device_id]
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3

        status = "normal"
        if used_gb >= self.memory_thresholds[device_id]["critical"]:
            status = "critical"
            logger.critical(f"🚨 GPU {device_id} memory critical: {used_gb:.2f}GB")
        elif used_gb >= self.memory_thresholds[device_id]["warning"]:
            status = "warning"
            logger.warning(f"⚠️  GPU {device_id} memory warning: {used_gb:.2f}GB")

        return {
            "device": device_id,
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "usage_percent": round(used_gb / total_gb * 100, 1),
            "status": status
        }

    def clear_cache(self, device_id: int):
        """Clear GPU cache for specified device"""
        try:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
            logger.info(f"✅ GPU {device_id} cache cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear GPU {device_id} cache: {e}")

    @staticmethod
    def check_system_memory():
        """Check system memory usage (NEW in v3)"""
        mem = psutil.virtual_memory()
        if mem.percent > 85:
            logger.critical(f"🚨 System memory critical: {mem.percent}%")
            return {"status": "critical", "percent": mem.percent}
        elif mem.percent > 75:
            logger.warning(f"⚠️  System memory warning: {mem.percent}%")
            return {"status": "warning", "percent": mem.percent}
        return {"status": "normal", "percent": mem.percent}

    def get_status(self):
        """Get all GPU and system status"""
        status = {
            "system": self.check_system_memory(),
            "gpus": []
        }

        for i in range(len(self.gpu_handles)):
            status["gpus"].append(self.check_memory(i))

        return status
