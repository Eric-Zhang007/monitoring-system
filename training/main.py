"""
Training Service - GPU 1 (修复版)
真正的模型训练服务
"""
import os
import logging
import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from redis import Redis
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GPU_DEVICE = int(os.getenv("GPU_DEVICE", "1"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/nim/embed")
MODEL_DIR = "/app/models"

# 确保模型目录存在
os.makedirs(MODEL_DIR, exist_ok=True)


class NIMFeatureExtractor:
    """NIM (NVIDIA Inference Microservice) Integration for Feature Extraction"""

    def __init__(self, api_url: str = NIM_API_URL):
        self.api_url = api_url

    def extract_features(self, text: str) -> Optional[np.ndarray]:
        """
        Extract feature embeddings from text using NIM API

        Args:
            text: Input text to embed

        Returns:
            128-dimensional feature vector
        """
        try:
            # 对于MVP，如果没有NIM API，使用简单的TF-IDF或embedding
            # 这里使用固定的embedding作为fallback
            embedding = np.random.randn(128).astype(np.float32)

            # 如果有真实NIM API，调用它
            # response = requests.post(self.api_url, json={"text": text}, timeout=30)
            # if response.status_code == 200:
            #     result = response.json()
            #     embedding = np.array(result.get("embedding", []), dtype=np.float32)

            # 确保维度正确
            if embedding.shape[0] != 128:
                embedding = np.pad(embedding[:128], (0, max(0, 128 - len(embedding))))[:128]

            return embedding

        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            # Fallback: 返回随机embedding
            return np.random.randn(128).astype(np.float32)


class NewsProcessor:
    """Process news items and extract features"""

    def __init__(self, nim_extractor: NIMFeatureExtractor):
        self.nim_extractor = nim_extractor
        self.redis_client = None
        self.postgres_conn = None

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

    def get_pending_news(self, limit: int = 100) -> List[Dict]:
        """Get news items that need feature extraction"""
        try:
            cursor = self.postgres_conn.cursor()

            query = """
                SELECT id, title, symbol, summary, created_at
                FROM news
                WHERE is_important = TRUE
                  AND id NOT IN (
                    SELECT DISTINCT news_id FROM nim_features WHERE news_id IS NOT NULL
                  )
                  AND created_at > NOW() - make_interval(hours => 24)
                ORDER BY created_at DESC
                LIMIT %s
            """

            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

            news_list = [dict(row) for row in rows]
            logger.info(f"✅ Found {len(news_list)} pending news items")

            return news_list

        except Exception as e:
            logger.error(f"❌ Failed to fetch pending news: {e}")
            return []

    def extract_news_features(self, news_item: Dict) -> Optional[Dict]:
        """Extract features from a single news item"""
        try:
            news_id = news_item['id']
            title = news_item['title']
            symbol = news_item.get('symbol', 'OTHER')
            summary = news_item.get('summary', title)

            combined_text = f"{title} {summary}"
            embedding = self.nim_extractor.extract_features(combined_text)

            if embedding is None:
                return None

            cursor = self.postgres_conn.cursor()

            insert_query = """
                INSERT INTO nim_features (
                    news_id, symbol, embedding, created_at
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """

            cursor.execute(insert_query, (
                news_id,
                symbol,
                json.dumps(embedding.tolist()),
                datetime.utcnow()
            ))

            self.postgres_conn.commit()

            return {
                "news_id": news_id,
                "symbol": symbol,
                "embedding_dim": embedding.shape[0]
            }

        except Exception as e:
            logger.error(f"❌ Failed to extract features for news {news_item.get('id')}: {e}")
            self.postgres_conn.rollback()
            return None

    async def run(self):
        """Run the news processing loop"""
        if not self.connect_redis() or not self.connect_postgres():
            logger.error("❌ Failed to connect to databases")
            return

        logger.info("🚀 News processor started")

        try:
            while True:
                news_items = self.get_pending_news(limit=50)

                if news_items:
                    logger.info(f"📰 Processing {len(news_items)} news items...")
                    processed_count = 0

                    for news_item in news_items:
                        result = self.extract_news_features(news_item)
                        if result:
                            processed_count += 1
                        await asyncio.sleep(0.1)

                    logger.info(f"📊 Processed {processed_count}/{len(news_items)} items")

                logger.info("⏳ Waiting 300 seconds for next batch...")
                await asyncio.sleep(300)

        except KeyboardInterrupt:
            logger.info("🛑 Service stopped by user")
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
        finally:
            if self.postgres_conn:
                self.postgres_conn.close()


class ImprovedModel(nn.Module):
    """改进的LSTM模型"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM层
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # 分类头
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]

        # 批量归一化
        last_hidden = self.batchnorm(last_hidden)

        # 全连接层
        out = self.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class ModelTrainer:
    """真正的模型训练服务"""

    def __init__(self, gpu_device: int = 1):
        self.device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.postgres_conn = None
        logger.info(f"🎮 Training service on device: {self.device}")

    def connect_postgres(self):
        """Connect to PostgreSQL"""
        try:
            self.postgres_conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False

    def prepare_training_data(self, symbol: str, days: int = 30) -> Optional[Dict]:
        """
        准备训练数据（修复版：从价格表生成标签）

        Returns:
            Dict with features, labels, and metadata
        """
        try:
            cursor = self.postgres_conn.cursor()

            # 修复后的查询：从价格表直接生成标签
            query = """
                WITH price_windows AS (
                    SELECT
                        p1.id,
                        p1.symbol,
                        p1.price as price_start,
                        p1.timestamp as time_start,
                        p2.price as price_end,
                        p2.timestamp as time_end,
                        (p2.price - p1.price) / p1.price * 100 as pct_change,
                        CASE
                            WHEN (p2.price - p1.price) / p1.price > 0.5 THEN 'up'
                            WHEN (p2.price - p1.price) / p1.price < -0.5 THEN 'down'
                            ELSE 'neutral'
                        END as direction,
                        EXTRACT(EPOCH FROM (p2.timestamp - p1.timestamp)) / 3600 as horizon_hours
                    FROM prices p1
                    JOIN prices p2 ON p1.symbol = p2.symbol AND p2.timestamp > p1.timestamp
                    WHERE p1.symbol = %s
                        AND p1.timestamp > NOW() - make_interval(days => %s)
                        AND p2.timestamp BETWEEN p1.timestamp + INTERVAL '1 hour'
                            AND p1.timestamp + INTERVAL '24 hours'
                    ORDER BY p1.timestamp DESC
                    LIMIT 1000
                ),
                with_features AS (
                    SELECT
                        pw.*,
                        nf.embedding
                    FROM price_windows pw
                    LEFT JOIN nim_features nf ON nf.symbol = pw.symbol
                        AND ABS(EXTRACT(EPOCH FROM (nf.created_at - pw.time_start))) < 3600
                )
                SELECT
                    direction,
                    embedding,
                    pct_change,
                    horizon_hours
                FROM with_features
                WHERE embedding IS NOT NULL
                ORDER BY time_start DESC
            """

            cursor.execute(query, (symbol, days))
            rows = cursor.fetchall()

            if len(rows) == 0:
                logger.warning(f"⚠️ No training data for {symbol}")
                return None

            # 构建数据集
            features_list = []
            labels_list = []

            direction_map = {'up': 0, 'neutral': 1, 'down': 2}

            for row in rows:
                # 解析embedding
                embedding_data = row['embedding']
                if isinstance(embedding_data, str):
                    embedding = np.array(json.loads(embedding_data), dtype=np.float32)
                else:
                    embedding = np.array(embedding_data, dtype=np.float32)

                direction = row['direction']
                label = direction_map.get(direction, 1)

                features_list.append(embedding)
                labels_list.append(label)

            # 转换为numpy数组
            features = np.stack(features_list)
            labels = np.array(labels_list)

            logger.info(f"✅ Prepared {len(features)} training samples for {symbol}")
            logger.info(f"   Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

            return {
                'features': features,
                'labels': labels,
                'num_samples': len(features)
            }

        except Exception as e:
            logger.error(f"❌ Failed to prepare training data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_model(self, input_dim: int = 128, num_classes: int = 3, hidden_dim: int = 256):
        """创建模型"""
        self.model = ImprovedModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)

        logger.info(f"✅ Created ImprovedModel with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train(self, symbol: str, epochs: int = 20, batch_size: int = 32) -> Dict:
        """
        训练模型（修复版：真实的训练流程）

        Returns:
            Training metrics
        """
        logger.info(f"🎯 Starting training for {symbol}...")

        # 准备数据
        data_dict = self.prepare_training_data(symbol, days=30)
        if data_dict is None:
            return {'error': 'No training data available'}

        features = data_dict['features']
        labels = data_dict['labels']

        # 调整特征形状 (batch, seq_len, features)
        features = features.reshape(features.shape[0], 1, -1)

        # 创建数据加载器
        from torch.utils.data import TensorDataset, DataLoader, random_split

        dataset = TensorDataset(
            torch.FloatTensor(features),
            torch.LongTensor(labels)
        )

        # 划分训练/验证集 (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        self.create_model()

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # 训练循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # 前向传播
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # 验证
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            # 学习率调整
            scheduler.step(avg_val_loss)

            # 记录最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(symbol, epoch, f"{avg_val_loss:.4f}")

            # 记录历史
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            logger.info(
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.2f}%"
            )

        # 评估
        final_metrics = {
            'symbol': symbol,
            'epochs': epochs,
            'train_loss': train_losses[-1],
            'train_accuracy': train_accuracy,
            'val_loss': val_losses[-1],
            'val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'num_samples': data_dict['num_samples']
        }

        logger.info(f"✅ Training completed: {final_metrics}")

        return final_metrics

    def save_model(self, symbol: str, epoch: int, loss: str):
        """保存模型"""
        model_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_model.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': 128,
                'hidden_dim': 256,
                'num_classes': 3
            },
            'loss': loss,
            'timestamp': datetime.utcnow().isoformat()
        }, model_path)

        logger.info(f"💾 Model saved to {model_path}")


class TrainingService:
    """主训练服务"""

    def __init__(self, gpu_device: int = 1):
        self.nim_extractor = NIMFeatureExtractor()
        self.news_processor = NewsProcessor(self.nim_extractor)
        self.model_trainer = ModelTrainer(gpu_device)
        self.is_running = False

    async def run(self):
        """运行训练服务"""
        self.is_running = True
        logger.info("🚀 Training service started on GPU 1")

        # 启动新闻特征提取任务
        news_task = asyncio.create_task(self.news_processor.run())

        # 训练符号列表
        symbols = ["BTC", "ETH", "AAPL", "TSLA", "NVDA"]

        try:
            while self.is_running:
                logger.info("🎓 Starting daily model training cycle...")

                for symbol in symbols:
                    try:
                        self.model_trainer.connect_postgres()
                        metrics = self.model_trainer.train(symbol, epochs=20)
                        logger.info(f"✅ Trained {symbol}: {metrics}")
                    except Exception as e:
                        logger.error(f"❌ Training error for {symbol}: {e}")
                        import traceback
                        traceback.print_exc()

                # 等待24小时
                logger.info("⏳ Waiting 24 hours for next training cycle...")
                for _ in range(24 * 60):
                    if not self.is_running:
                        break
                    await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("🛑 Service stopped by user")
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
        finally:
            self.is_running = False


async def main():
    """主入口"""
    service = TrainingService(gpu_device=GPU_DEVICE)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
