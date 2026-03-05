# 部署快速指南（Clone 后最少步骤）

## 1. 环境

1. Python：`3.12.x`
2. PostgreSQL：可连通，`DATABASE_URL` 有效
3. PostgreSQL 扩展：需安装 `pgvector`（`vector` 扩展）
3. 执行：
```bash
bash scripts/server_quickstart.sh
```

该脚本会：
1. 创建 `.venv`
2. 安装 runtime/train/dev 依赖
3. 若 `.env` 缺失则从 `.env.example` 复制

## 2. 初始化数据库

```bash
python - <<'PY'
import os
from pathlib import Path
import psycopg2
dsn = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(Path("scripts/init_db.sql").read_text(encoding="utf-8"))
    conn.commit()
print("init_db_ok")
PY
```

若初始化时报 `extension "vector" is not available`：
```bash
sudo apt-get update
sudo apt-get install -y postgresql-16-pgvector
sudo service postgresql restart
```

## 3. 准备文本编码模型

```bash
python scripts/setup_text_encoder.py --model-id intfloat/multilingual-e5-small --out-dir artifacts/models/text_encoder/multilingual-e5-small
export TEXT_EMBED_MODEL_PATH=artifacts/models/text_encoder/multilingual-e5-small
```

## 4. 构建训练输入

```bash
python scripts/build_feature_store.py --start 2018-01-01T00:00:00Z
python scripts/build_text_embeddings.py --start 2018-01-01T00:00:00Z --model-path "$TEXT_EMBED_MODEL_PATH"
python scripts/merge_feature_views.py --start 2018-01-01T00:00:00Z
```

可选：先补齐全历史多分辨率 K 线并生成对齐上下文（供 Agent/风控按需读取）：
```bash
bash scripts/collect_all_timeframe_market_bars.sh
```

## 5. 训练与推理

```bash
python training/train_liquid.py --model-id liquid_main --out-dir artifacts/models/liquid_main
python training/train_vc.py --model-id vc_main --out-dir artifacts/models/vc_main
python inference/main.py --symbol BTC
```

## 6. 验收

```bash
bash scripts/run_strict_e2e_acceptance.sh
```

## 常见失败点

1. `artifact_missing:*manifest.json`：未先训练或 `LIQUID_MODEL_DIR/VC_MODEL_DIR` 指向错误。
2. `schema_hash_mismatch_*`：特征契约与工件不一致，需要重建特征并重训。
3. 文本 embedding 目录缺失：`build_text_embeddings.py` 会 fail-fast，不会 fallback。
