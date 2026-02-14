"""
Inference Service - GPU 0
实时推理服务，提供金融价格预测
"""
import os
import logging
import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from redis import Redis
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GPU_DEVICE = int(os.getenv("GPU_DEVICE", "0"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")


class PricePredictor:
    """Price prediction model for real-time inference"""

    def __init__(self, gpu_device: int = 0):
        self.device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scenario_names = ["上涨", "盘整", "下跌"]
        self.time_horizons = ["1h", "1d", "7d"]
        logger.info(f"🎮 Initialised predictor on device: {self.device}")

    def load_model(self, model_path: str = "/app/models/lstm_model.pth"):
        """Load the trained model"""
        try:
            # For MVP, we'll create a lightweight LSTM model
            # In production, load from saved checkpoint
            self.model = torch.nn.LSTM(
                input_size=128,  # Feature dimension (text embeddings from NIM)
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            ).to(self.device)

            # Load weights if exists
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ Loaded model from {model_path}")
            else:
                logger.warning(f"⚠️ Model file not found, using initialized weights")

            self.model.eval()
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def predict(
        self,
        features: np.ndarray,
        current_price: float
    ) -> Dict:
        """
        Generate predictions for a financial instrument

        Args:
            features: Feature vector from NIM (128-dim embedding)
            current_price: Current price of the instrument

        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            # Convert to tensor
            if features.ndim == 1:
                features = features.reshape(1, 1, -1)
            elif features.ndim == 2:
                # Shape: (batch, seq_len, features)
                pass

            features_tensor = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                # Get model output (logits)
                output, _ = self.model(features_tensor)

                # Get last hidden state
                last_output = output[:, -1, :]

                # Scenario classification (linear layer)
                scenario_logits = torch.nn.functional.linear(
                    last_output,
                    torch.randn(3, 256).to(self.device)  # Temporary weights
                )

                # Convert to probabilities
                scenario_probs = torch.softmax(scenario_logits, dim=-1)

                # Confidence score (max probability)
                confidence_score, predicted_scenario = torch.max(scenario_probs, dim=-1)

                # Expected price change based on scenario
                # This is a simplified calculation for MVP
                predicted_scenario_idx = predicted_scenario.item()
                confident_scenario = self.scenario_names[predicted_scenario_idx]

                # Price change ranges based on scenario
                if confident_scenario == "上涨":
                    change_range = (0.005, 0.035)  # 0.5% ~ 3.5%
                    expected_direction = "up"
                elif confident_scenario == "盘整":
                    change_range = (-0.008, 0.008)  # -0.8% ~ 0.8%
                    expected_direction = "neutral"
                else:  # 下跌
                    change_range = (-0.035, -0.005)  # -3.5% ~ -0.5%
                    expected_direction = "down"

                # Sample from uniform distribution within range
                np_rng = np.random.default_rng()
                expected_change_pct = np_rng.uniform(*change_range)

                expected_price = current_price * (1 + expected_change_pct)

            # Build result
            result = {
                "scenario": confident_scenario,
                "direction": expected_direction,
                "confidence": confidence_score.item(),
                "expected_change_pct": expected_change_pct,
                "expected_price": expected_price,
                "scenario_probabilities": {
                    "up": scenario_probs[0, 0].item(),
                    "neutral": scenario_probs[0, 1].item(),
                    "down": scenario_probs[0, 2].item()
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None


class InferenceService:
    """Main inference service"""

    def __init__(self, gpu_device: int = 0):
        self.predictor = PricePredictor(gpu_device)
        self.redis_client = None
        self.postgres_conn = None
        self.is_running = False

    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False

    def connect_postgres(self):
        """Connect to PostgreSQL"""
        try:
            self.postgres_conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False

    def get_latest_features(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get latest feature vector from PostgreSQL

        Query uses parameterized intervals to prevent SQL injection
        """
        try:
            cursor = self.postgres_conn.cursor()

            # Parameterized query with make_interval
            query = """
                SELECT embedding, created_at FROM nim_features
                WHERE symbol = %s AND created_at > NOW() - make_interval(hours => 1)
                ORDER BY created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (symbol,))
            row = cursor.fetchone()

            if row:
                # Parse embedding (stored as JSON array or binary)
                embedding_data = row['embedding']

                # Handle both JSON and binary formats
                if isinstance(embedding_data, str):
                    embedding = np.array(json.loads(embedding_data), dtype=np.float32)
                else:
                    # Assume it's already a list or array
                    embedding = np.array(embedding_data, dtype=np.float32)

                logger.info(f"✅ Retrieved features for {symbol}")
                return embedding
            else:
                logger.warning(f"⚠️ No features found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to retrieve features: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Redis or cache"""
        try:
            key = f"price:{symbol}"
            price_str = self.redis_client.get(key)

            if price_str:
                price_data = json.loads(price_str)
                return float(price_data.get("price", 0))
            else:
                logger.warning(f"⚠️ No price found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to get price: {e}")
            return None

    def save_prediction(self, symbol: str, prediction: Dict):
        """Save prediction to PostgreSQL"""
        try:
            cursor = self.postgres_conn.cursor()

            insert_query = """
                INSERT INTO predictions (
                    symbol, scenario, direction, confidence,
                    expected_change_pct, expected_price,
                    scenario_probabilities, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            cursor.execute(insert_query, (
                symbol,
                prediction['scenario'],
                prediction['direction'],
                prediction['confidence'],
                prediction['expected_change_pct'],
                prediction['expected_price'],
                json.dumps(prediction['scenario_probabilities']),
                prediction['timestamp']
            ))

            self.postgres_conn.commit()
            logger.info(f"✅ Saved prediction for {symbol}")

        except Exception as e:
            logger.error(f"❌ Failed to save prediction: {e}")
            self.postgres_conn.rollback()

    def publish_prediction(self, symbol: str, prediction: Dict):
        """Publish prediction to Redis Streams"""
        try:
            stream_name = "prediction_stream"

            self.redis_client.xadd(
                stream_name,
                {
                    "symbol": symbol,
                    "scenario": prediction['scenario'],
                    "direction": prediction['direction'],
                    "confidence": str(prediction['confidence']),
                    "expected_change_pct": str(prediction['expected_change_pct']),
                    "expected_price": str(prediction['expected_price']),
                    "scenario_probabilities": json.dumps(prediction['scenario_probabilities']),
                    "timestamp": prediction['timestamp']
                }
            )

            logger.info(f"✅ Published prediction for {symbol} to stream")

        except Exception as e:
            logger.error(f"❌ Failed to publish prediction: {e}")

    async def process_symbol(self, symbol: str):
        """Process prediction for a single symbol"""
        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            logger.warning(f"⚠️ No price available for {symbol}, skipping")
            return

        # Get features
        features = self.get_latest_features(symbol)
        if features is None:
            logger.warning(f"⚠️ No features available for {symbol}, skipping")
            return

        # Generate prediction
        prediction = self.predictor.predict(features, current_price)
        if prediction:
            # Save to database
            self.save_prediction(symbol, prediction)

            # Publish to stream
            self.publish_prediction(symbol, prediction)

            logger.info(f"🎯 {symbol} - {prediction['scenario']} (confidence: {prediction['confidence']:.3f})")

    async def run(self):
        """Run the inference service"""
        # Load model
        if not self.predictor.load_model():
            logger.error("❌ Failed to load model, exiting")
            return

        # Connect to databases
        if not self.connect_redis() or not self.connect_postgres():
            logger.error("❌ Failed to connect to databases, exiting")
            return

        self.is_running = True
        logger.info("🚀 Inference service started")

        # Symbols to monitor (in production, fetch from database)
        symbols = ["BTC", "ETH", "AAPL", "TSLA", "NVDA"]

        try:
            while self.is_running:
                for symbol in symbols:
                    try:
                        await self.process_symbol(symbol)
                        await asyncio.sleep(0.1)  # Small delay between symbols
                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {e}")

                # Wait before next round
                logger.info("⏳ Waiting 60 seconds for next batch...")
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("🛑 Service stopped by user")
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.postgres_conn:
            self.postgres_conn.close()
        logger.info("✅ Service cleaned up")


async def main():
    """Main entry point"""
    service = InferenceService(gpu_device=GPU_DEVICE)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
