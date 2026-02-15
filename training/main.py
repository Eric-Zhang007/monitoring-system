"""
Training Service V2
- Split pipelines for VC and Liquid tracks
- Registers model metadata in model_registry
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - runtime dependency check is handled by launcher
    torch = None
    dist = None

from feature_pipeline import FeaturePipeline
from liquid_model_trainer import LiquidModelTrainer
from vc_model_trainer import VCModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYMBOLS = os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK").split(",")
TRAIN_INTERVAL_SEC = int(os.getenv("TRAIN_INTERVAL_SEC", "21600"))
TRAIN_RUN_ONCE = os.getenv("TRAIN_RUN_ONCE", "0").lower() in {"1", "true", "yes", "y"}
TRAIN_ENABLE_VC = os.getenv("TRAIN_ENABLE_VC", "1").lower() in {"1", "true", "yes", "y"}
TRAIN_ENABLE_LIQUID = os.getenv("TRAIN_ENABLE_LIQUID", "1").lower() in {"1", "true", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


def _init_distributed() -> Dict[str, int]:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size <= 1:
        return {"rank": 0, "world_size": 1, "local_rank": 0}
    if torch is None or dist is None:
        raise RuntimeError("distributed_requested_but_torch_missing")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        init_kwargs: Dict[str, Any] = {}
        if backend == "nccl" and local_rank >= 0:
            init_kwargs["device_id"] = local_rank
        try:
            dist.init_process_group(backend=backend, init_method="env://", **init_kwargs)
        except TypeError:
            dist.init_process_group(backend=backend, init_method="env://")
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def _is_distributed() -> bool:
    return bool(dist is not None and dist.is_available() and dist.is_initialized())


class TrainingServiceV2:
    def __init__(self, *, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.pipeline = FeaturePipeline()
        self.vc_trainer = VCModelTrainer(
            self.pipeline,
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )
        self.liquid_trainer = LiquidModelTrainer(
            self.pipeline,
            symbols=[s.strip().upper() for s in SYMBOLS if s.strip()],
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )

    def _barrier(self) -> None:
        if _is_distributed():
            if torch is not None and torch.cuda.is_available() and self.local_rank >= 0:
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def _gather_dict(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not _is_distributed():
            return [payload]
        world_size = dist.get_world_size()
        gathered: List[Dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, payload)
        return [x for x in gathered if isinstance(x, dict)]

    async def run_once(self):
        if TRAIN_ENABLE_VC and self.rank == 0:
            vc = self.vc_trainer.train()
            logger.info("vc_train=%s", vc)
        elif TRAIN_ENABLE_VC:
            logger.info("vc_train=skipped_non_primary_rank rank=%s", self.rank)
        self._barrier()
        if TRAIN_ENABLE_LIQUID:
            local_result = self.liquid_trainer.train_all()
            gathered = self._gather_dict(local_result)
            if self.rank == 0:
                merged_results: List[Dict[str, Any]] = []
                for item in gathered:
                    merged_results.extend(item.get("results", []))
                logger.info(
                    "liquid_train=%s",
                    {
                        "status": "ok",
                        "world_size": self.world_size,
                        "results": merged_results,
                    },
                )
        self._barrier()

    async def run(self):
        logger.info(
            "training-v2 started interval=%ss run_once=%s enable_vc=%s enable_liquid=%s rank=%s world_size=%s local_rank=%s",
            TRAIN_INTERVAL_SEC,
            TRAIN_RUN_ONCE,
            TRAIN_ENABLE_VC,
            TRAIN_ENABLE_LIQUID,
            self.rank,
            self.world_size,
            self.local_rank,
        )
        if TRAIN_RUN_ONCE:
            await self.run_once()
            if self.rank == 0:
                logger.info("training-v2 run_once completed")
            return
        while True:
            try:
                await self.run_once()
                await asyncio.sleep(TRAIN_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("training-v2 stopped")
                break
            except Exception as exc:
                logger.error("training-v2 cycle failed: %s", exc)
                await asyncio.sleep(60)


if __name__ == "__main__":
    dist_ctx = _init_distributed()
    try:
        asyncio.run(
            TrainingServiceV2(
                rank=dist_ctx["rank"],
                world_size=dist_ctx["world_size"],
                local_rank=dist_ctx["local_rank"],
            ).run()
        )
    finally:
        if _is_distributed():
            dist.destroy_process_group()
