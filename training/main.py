"""
Training Service V2
- Split pipelines for VC and Liquid tracks
- Registers model metadata in model_registry
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
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

SYMBOLS = os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL").split(",")
TRAIN_INTERVAL_SEC = int(os.getenv("TRAIN_INTERVAL_SEC", "21600"))
TRAIN_RUN_ONCE = os.getenv("TRAIN_RUN_ONCE", "0").lower() in {"1", "true", "yes", "y"}
TRAIN_ENABLE_VC = os.getenv("TRAIN_ENABLE_VC", "1").lower() in {"1", "true", "yes", "y"}
TRAIN_ENABLE_LIQUID = os.getenv("TRAIN_ENABLE_LIQUID", "1").lower() in {"1", "true", "yes", "y"}
TRAIN_ENABLE_BACKBONE_EXPERIMENTS = os.getenv("TRAIN_ENABLE_BACKBONE_EXPERIMENTS", "0").lower() in {"1", "true", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


LIQUID_TRAIN_START = str(os.getenv("LIQUID_TRAIN_START", "")).strip()
LIQUID_TRAIN_END = str(os.getenv("LIQUID_TRAIN_END", "")).strip()
LIQUID_TRAIN_LIMIT = _env_int("LIQUID_TRAIN_LIMIT", 4000)
LIQUID_TRAIN_MAX_SAMPLES = _env_int("LIQUID_TRAIN_MAX_SAMPLES", 0)
LIQUID_TRAIN_SAMPLE_MODE = str(os.getenv("LIQUID_TRAIN_SAMPLE_MODE", "uniform")).strip().lower() or "uniform"
LIQUID_TRAIN_MODE = str(os.getenv("LIQUID_TRAIN_MODE", "production")).strip().lower() or "production"
LIQUID_TRAIN_LOOKBACK_DAYS = _env_int("LIQUID_TRAIN_LOOKBACK_DAYS", 365)
LIQUID_DATA_MODE = str(os.getenv("LIQUID_DATA_MODE", "production")).strip().lower() or "production"


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
    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        liquid_train_start: str = "",
        liquid_train_end: str = "",
        liquid_train_limit: int = 4000,
        liquid_train_max_samples: int = 0,
        liquid_train_sample_mode: str = "uniform",
        liquid_data_mode: str = "production",
    ):
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
            train_start=str(liquid_train_start or "").strip(),
            train_end=str(liquid_train_end or "").strip(),
            train_limit=max(0, int(liquid_train_limit)),
            train_max_samples=max(0, int(liquid_train_max_samples)),
            train_sample_mode=str(liquid_train_sample_mode or "uniform").strip().lower() or "uniform",
            train_data_mode=str(liquid_data_mode or "production").strip().lower() or "production",
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
        if TRAIN_ENABLE_BACKBONE_EXPERIMENTS and self.rank == 0:
            try:
                from backbone_experiments import run_experiment_suite_from_env

                exp_out = run_experiment_suite_from_env()
                logger.info(
                    "liquid_backbone_experiments=%s",
                    {
                        "status": exp_out.get("status"),
                        "rows": (exp_out.get("dataset") or {}).get("rows", 0),
                        "ready_backbones": exp_out.get("ready_backbones", []),
                        "torch_available": exp_out.get("torch_available", False),
                    },
                )
            except Exception as exc:
                logger.error("liquid_backbone_experiments_failed: %s", exc)
        self._barrier()

    async def run(self):
        logger.info(
            "training-v2 started interval=%ss run_once=%s enable_vc=%s enable_liquid=%s enable_backbone_exp=%s rank=%s world_size=%s local_rank=%s liquid_data_mode=%s liquid_train_mode=%s liquid_lookback_days=%s liquid_start=%s liquid_end=%s liquid_limit=%s liquid_max_samples=%s liquid_sample_mode=%s",
            TRAIN_INTERVAL_SEC,
            TRAIN_RUN_ONCE,
            TRAIN_ENABLE_VC,
            TRAIN_ENABLE_LIQUID,
            TRAIN_ENABLE_BACKBONE_EXPERIMENTS,
            self.rank,
            self.world_size,
            self.local_rank,
            getattr(self.liquid_trainer, "train_data_mode", "production"),
            getattr(self.liquid_trainer, "train_mode", "production"),
            getattr(self.liquid_trainer, "train_lookback_days", 0),
            (getattr(self.liquid_trainer, "train_start", None) or ""),
            (getattr(self.liquid_trainer, "train_end", None) or ""),
            getattr(self.liquid_trainer, "train_limit", 0),
            getattr(self.liquid_trainer, "train_max_samples", 0),
            getattr(self.liquid_trainer, "train_sample_mode", "uniform"),
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
    ap = argparse.ArgumentParser(description="Training Service V2 entrypoint")
    ap.add_argument("--liquid-start", default=LIQUID_TRAIN_START)
    ap.add_argument("--liquid-end", default=LIQUID_TRAIN_END)
    ap.add_argument("--liquid-limit", type=int, default=LIQUID_TRAIN_LIMIT)
    ap.add_argument("--liquid-max-samples", type=int, default=LIQUID_TRAIN_MAX_SAMPLES)
    ap.add_argument("--liquid-sample-mode", default=LIQUID_TRAIN_SAMPLE_MODE)
    ap.add_argument("--liquid-train-mode", default=LIQUID_TRAIN_MODE, choices=["production", "fast"])
    ap.add_argument("--liquid-lookback-days", type=int, default=LIQUID_TRAIN_LOOKBACK_DAYS)
    ap.add_argument("--liquid-data-mode", default=LIQUID_DATA_MODE, choices=["production", "research"])
    cli_args = ap.parse_args()

    # Basic format guard for CLI datetimes to fail fast.
    for raw in (cli_args.liquid_start, cli_args.liquid_end):
        text = str(raw or "").strip()
        if not text:
            continue
        norm = text.replace(" ", "T")
        if norm.endswith("Z"):
            norm = norm[:-1] + "+00:00"
        dt = datetime.fromisoformat(norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

    dist_ctx = _init_distributed()
    try:
        os.environ["LIQUID_TRAIN_MODE"] = str(cli_args.liquid_train_mode or LIQUID_TRAIN_MODE)
        os.environ["LIQUID_TRAIN_LOOKBACK_DAYS"] = str(max(1, int(cli_args.liquid_lookback_days)))
        asyncio.run(
            TrainingServiceV2(
                rank=dist_ctx["rank"],
                world_size=dist_ctx["world_size"],
                local_rank=dist_ctx["local_rank"],
                liquid_train_start=str(cli_args.liquid_start or "").strip(),
                liquid_train_end=str(cli_args.liquid_end or "").strip(),
                liquid_train_limit=max(0, int(cli_args.liquid_limit)),
                liquid_train_max_samples=max(0, int(cli_args.liquid_max_samples)),
                liquid_train_sample_mode=str(cli_args.liquid_sample_mode or "uniform").strip().lower() or "uniform",
                liquid_data_mode=str(cli_args.liquid_data_mode or "production").strip().lower() or "production",
            ).run()
        )
    finally:
        if _is_distributed():
            dist.destroy_process_group()
