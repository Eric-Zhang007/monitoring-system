"""
Training Service V2
- Split pipelines for VC and Liquid tracks
- Registers model metadata in model_registry
"""
from __future__ import annotations

import asyncio
import logging
import os

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


class TrainingServiceV2:
    def __init__(self):
        self.pipeline = FeaturePipeline()
        self.vc_trainer = VCModelTrainer(self.pipeline)
        self.liquid_trainer = LiquidModelTrainer(self.pipeline, symbols=[s.strip().upper() for s in SYMBOLS if s.strip()])

    async def run_once(self):
        if TRAIN_ENABLE_VC:
            vc = self.vc_trainer.train()
            logger.info("vc_train=%s", vc)
        if TRAIN_ENABLE_LIQUID:
            liquid = self.liquid_trainer.train_all()
            logger.info("liquid_train=%s", liquid)

    async def run(self):
        logger.info(
            "training-v2 started interval=%ss run_once=%s enable_vc=%s enable_liquid=%s",
            TRAIN_INTERVAL_SEC,
            TRAIN_RUN_ONCE,
            TRAIN_ENABLE_VC,
            TRAIN_ENABLE_LIQUID,
        )
        if TRAIN_RUN_ONCE:
            await self.run_once()
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
    asyncio.run(TrainingServiceV2().run())
