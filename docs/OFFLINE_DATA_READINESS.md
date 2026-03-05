# Offline Data Readiness (Strict)

## 目标
为 `training/train_liquid.py` 提供可训练、可追溯、可解释的离线数据门禁。严格模式下 gate 失败即阻断训练。

## 数据依赖来源（由代码扫描归纳）
审计脚本 `scripts/audit_offline_training_data.py` 会扫描以下文件中的 SQL/表名引用：

- `training/datasets/liquid_sequence_dataset.py`
- `features/sequence.py`
- `scripts/build_feature_store.py`
- `scripts/build_text_embeddings.py`
- `scripts/merge_feature_views.py`
- `training/train_liquid.py`

### 必需表
- `market_bars`
- `feature_snapshots_main`
- `feature_matrix_main`

### 文本源（至少其一存在）
- `social_posts_raw`
- `social_comments_raw`
- `events`

### 可选增强表
- `orderbook_l2`
- `funding_rates`
- `onchain_signals`
- `social_text_embeddings`（缺失不会必然 hard-fail，但会影响多模态贡献）

## 审计脚本
```bash
python scripts/audit_offline_training_data.py \
  --database-url "$DATABASE_URL" \
  --track liquid \
  --symbols BTC,ETH,SOL \
  --start 2018-01-01T00:00:00Z \
  --end 2026-02-24T00:00:00Z \
  --lookback 96 \
  --bucket 5m
```

输出会写入 DB 表 `offline_data_audits`，并返回 JSON：

- `gates.ready`: 训练是否可放行
- `gates.reasons`: 阻断原因（strict）
- `coverage.market_bars`: 每 symbol 覆盖率/缺桶率/最长连续窗口
- `coverage.feature_snapshots_main`: 行数、schema/hash/dim 一致性
- `coverage.feature_matrix_main`: values/mask 维度正确率、schema mismatch 次数
- `coverage.text_source`: 文本源存在性、行数、embedding 覆盖率
- `freshness`: 各关键表最新时间与滞后秒数
- `label_ready`: 每 symbol 是否满足 `lookback + max_horizon`
- `universe_ready`: `asset_universe_snapshots` 可解析性 + 训练报告 universe 追踪能力
- `repair_suggestions`: 缺口修复建议

## 默认阈值
- `market_bars present_bucket_ratio >= 0.98`
- `feature_matrix_main present_bucket_ratio >= 0.95`
- `schema_hash_mismatch == 0`（strict）
- 文本覆盖默认阈值 `0.05`，默认仅告警（可通过 `AUDIT_ENFORCE_TEXT_COVERAGE=1` 升级为 hard gate）

## 指标解释
- `present_bucket_ratio`: 窗口内有数据桶 / 期望桶数。越接近 1 越好。
- `schema_hash_mismatch`: 训练契约不一致次数。非零意味着训练/推理 parity 风险。
- `vector_dim_ok_rows`: `values/mask` 与 `FEATURE_DIM` 完全匹配行数。低值表示特征产物损坏。
- `label_ready`: 若 false，说明标签未来窗口不够，训练样本会不稳定或被丢弃。
- `freshness.lag_seconds`: 数据新鲜度。线上推理应保持在可控时延内。

## API
- `GET /api/v2/audit/offline_data`: 最近一次审计结果
- `POST /api/v2/audit/offline_data/run`: 触发审计，返回 `task_id`
- `GET /api/v2/audit/offline_data/tasks/{task_id}`: 任务状态
- `GET /api/v2/audit/offline_data/stream/{task_id}`: SSE 进度流
