# Monitoring System (Strict)

生产主线为 strict-only，不保留灰度/回滚分支。核心契约：

1. Schema 单一真相源：`schema/liquid_feature_schema.yaml`
2. 禁止 pad/truncate；缺失仅通过 `mask` 表达
3. 主模型输入固定：`X_values[L,D] + X_mask[L,D]`
4. 工件必须包含：`manifest + weights + schema_snapshot + training_report`
5. 推理启动强校验，失败直接退出，无 silent fallback

## 快速开始（服务器）

1. 环境准备（推荐 Python 3.12）
```bash
bash scripts/server_quickstart.sh
```

2. 初始化数据库
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

3. 构建特征矩阵
```bash
python scripts/build_feature_store.py --start 2018-01-01T00:00:00Z
python scripts/setup_text_encoder.py --model-id intfloat/multilingual-e5-small --out-dir artifacts/models/text_encoder/multilingual-e5-small
export TEXT_EMBED_MODEL_PATH=artifacts/models/text_encoder/multilingual-e5-small
python scripts/build_text_embeddings.py --start 2018-01-01T00:00:00Z --model-path "$TEXT_EMBED_MODEL_PATH"
python scripts/merge_feature_views.py --start 2018-01-01T00:00:00Z
```

4. 训练与推理
```bash
python training/train_liquid.py --model-id liquid_main --out-dir artifacts/models/liquid_main
python training/train_vc.py --model-id vc_main --out-dir artifacts/models/vc_main
python inference/main.py --symbol BTC
```

5. 端到端验收
```bash
bash scripts/run_strict_e2e_acceptance.sh
```

## 文档

1. 规则：`docs/PRODUCTION_REFACTOR_RULES.md`
2. Pipeline：`docs/STRICT_PIPELINE_ZH.md`
3. 部署：`docs/DEPLOYMENT_QUICKSTART_ZH.md`

<!-- AUTO_STATUS_SNAPSHOT:BEGIN -->
### Auto Snapshot
- track: `liquid`
- score_source: `model`
- sharpe: `n/a`
- max_drawdown: `n/a`
- execution_reject_rate: `n/a`
- hard_passed: `false`
- parity_status: `unknown`
- parity_matched_targets: `0`
- parity_paper_filled_orders: `0`
<!-- AUTO_STATUS_SNAPSHOT:END -->
