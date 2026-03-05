# Monitoring System (Strict Production)

本仓库是一个 strict-only 的量化预测/执行系统，目标是 Top50 面板联合训练与生产推理执行。

## 1. 设计原则（硬约束）

- strict-only：不引入灰度/回退分支。
- fail-fast：关键依赖缺失（工件/schema/cache/密钥/数据）立即报错终止。
- 单一特征真相源：`schema/liquid_feature_schema.yaml`。
- 缺失仅用 `mask` 表达，禁止 pad/truncate/伪造 embedding。
- 训练/评估/推理/决策统一成本口径：`cost/cost_profile.py`。

---

## 2. 仓库结构（核心模块）

- `schema/`：特征 schema 与契约。
- `features/`：特征契约、对齐、序列读取。
- `training/`：数据集、缓存构建、损失、训练流程。
- `models/`：backbone + 分布头 + MoE 路由。
- `inference/`：离线/在线推理入口。
- `backend/`：API、信号生成、执行、风控、审计。
- `cost/`：统一成本口径。
- `scripts/`：数据采集、特征构建、审计、训练、评估、打包、验收脚本。
- `tests/`：单测/契约测试/集成 smoke。

---

## 3. 数据分层与构建链路

### 3.1 数据分层（从原始到模型输入）

1. 原始行情/辅助数据层（DB）
- `market_bars`：多分辨率 K 线。
- `funding_rates`：资金费率。
- `onchain_signals`：衍生指标/拥挤度代理。
- `orderbook_l2`：盘口/深度代理。
- `events`：事件文本。

2. 数值特征快照层
- `feature_snapshots_main`
- 由 `scripts/build_feature_store.py` 构建。

3. 文本嵌入层
- `social_text_embeddings`
- 由 `scripts/build_text_embeddings.py` 构建。

4. 主特征矩阵层（训练/推理统一读取）
- `feature_matrix_main`
- 由 `scripts/merge_feature_views.py` 合并：
  - 基础数值特征来自 `feature_snapshots_main`
  - 文本特征按 as-of 对齐来自 `social_text_embeddings`
- 分片执行脚本：`scripts/merge_feature_views_chunked.sh`

5. 多时间框上下文层
- `market_context_multi_tf`
- 由 `scripts/build_multi_timeframe_context.py` 构建，提供 `5m/15m/1h/4h/1d` 上下文。

6. 训练缓存层（无 per-sample DB 查询）
- `artifacts/cache/liquid_top50/`
- 由 `scripts/build_training_cache.py` 构建。

### 3.2 Universe 与 Top50

- 快照脚本：`scripts/update_universe_top50.py`（或 `update_universe_snapshot.py`）
- 快照表：`asset_universe_snapshots`
- 快照文件：`artifacts/universe/liquid_top50_snapshot.json`
- 支持 hysteresis 与可回放 hash。

### 3.3 全历史多分辨率采集

```bash
bash scripts/collect_all_timeframe_market_bars.sh
```

常用参数：
- `TIMEFRAMES`（默认 `1m,3m,5m,15m,30m,1h,4h,6h,12h,1d`）
- `START` / `END`
- `SYMBOLS` / `SYMBOL_MAP`
- `PRIMARY_TIMEFRAME`

---

## 4. `feature_matrix_main` 是什么

`feature_matrix_main` 是模型统一输入层，每行对应 `(symbol, as_of_ts)`，包含：

- `values`：特征向量（当前主线 `feature_dim=169`）
- `mask`：同维缺失掩码（缺失=1）
- `features`：按 key 展开的 `{value, missing}`
- `feature_dim` / `schema_hash` / `feature_version`
- `synthetic_ratio`

关键行为：
- 维度异常会审计并按阈值 fail-fast。
- schema hash 强校验。
- 文本缺失只体现在 mask，不伪造向量。

---

## 5. 模型目前如何“吃数据”

### 5.1 训练缓存结构

`build_training_cache.py` 产物：
- 每个 symbol 一个 `*.npz`（`values/mask/close/end_ts/regime_features/regime_mask/multi_tf_context/multi_tf_mask`）
- `index.npz`（`symbol_id, t_idx, end_ts`）
- `cache_manifest.json`
- `data_audit.json`

### 5.2 Dataset 样本（`training/datasets/liquid_panel_cache_dataset.py`）

每个样本返回：
- `x_values`: `[lookback, feature_dim]`
- `x_mask`: `[lookback, feature_dim]`
- `symbol_id` / `symbol`
- 标签：`y_net`, `y_raw`, `cost_bps`, `direction`（horizons: `1h/4h/1d/7d`）
- `regime_features`, `regime_mask`
- `multi_timeframe_context`, `multi_timeframe_mask`
- 辅助：`future_vol`, `drawdown_proxy`

### 5.3 当前默认时间口径（主训练）

- `lookback=2016`（5m 下约 7 天）
- horizons：`1h,4h,1d,7d`
- `sample_stride_buckets=3`
- 成本标签由统一 `cost_profile` 计算。

---

## 6. 模型结构（当前主线）

主模型：`models/liquid_model.py`
- 共享 backbone（跨资产面板）
- `SymbolEmbedding`
- `MultiHorizonDistHead`（分布输出 + MoE 路由）

### 6.1 三个 backbone 实现

1. PatchTST
- 文件：`models/backbones/patchtst.py`
- 默认构造参数：`d_model=128, n_layers=2, n_heads=4, patch_len=16`

2. iTransformer
- 文件：`models/backbones/itransformer.py`
- 默认构造参数：`d_model=128, n_layers=2, n_heads=4`

3. TFT
- 文件：`models/backbones/tft.py`
- 默认构造参数：`d_model=128, n_heads=4`
- 注意：当前实现里 LSTM 层数固定 1。

路由工厂：`models/backbones/registry.py`
- `patchtst` 透传全部参数
- `itransformer` 忽略 `patch_len`
- `tft` 忽略 `patch_len` 与 `n_layers`

### 6.2 分布头输出（`models/heads/dist_head.py`）

输出契约（`models/outputs.py`）：
- `mu`（各 horizon 期望收益）
- `log_sigma` / `sigma`
- `q`（quantiles）
- `direction_logit`
- `expert_weights`
- `regime_probs`
- `df`（Student-t tail 参数）
- `aux`（attention/router/load-balance/uncertainty 等）

### 6.3 MoE 专家与路由

- 专家集合：`trend / mean_reversion / liquidation_risk / neutral`
- 路由输入：`regime + symbol_ctx + quality`
- 支持 top-k sparse gating（默认 top-2）

---

## 7. 损失、评估、门禁

### 7.1 训练损失

`training/losses/trading_losses.py` 组合：
- Student-t NLL
- Quantile pinball
- Direction loss
- Router/load-balance/entropy regularizer
- horizon smoothness / vol monotonic regularizer

### 7.2 评估与门禁

- 评估脚本：`scripts/eval_liquid_top50.py` / `scripts/eval_top50.py`
- 关键输出：`metrics.json`, `stratified_metrics.json`, `calibration.json`, `router_report.json`, `pnl_timeseries.csv`
- Fail-fast 门禁：泄露、schema mismatch、router collapse、工件缺件、成本口径不一致等。

---

## 8. 成本与执行（统一口径）

统一模块：`cost/cost_profile.py`

覆盖：
- 训练标签扣费
- 离线评估扣费
- 推理决策扣费

成本分项：
- `fee_bps`
- `slippage_bps`
- `impact_bps`
- `funding_bps`
- `infra_bps`

---

## 9. 推理、决策、执行、审计

- 推理入口：`inference/main.py`
- 后端服务：`backend/liquid_model_service.py`, `backend/v2_router.py`
- 决策追踪：`DecisionTrace`（模型输出、成本分项、阈值、动作、agent输出）

执行风控能力已补齐（融合 Following 设计语义）：
- kill switch / safe mode / panic close
- WS->REST 行情降级联动
- stoploss 本地守护与保护性平仓
- unknown position 检测与安全降级

---

## 10. Financial Analysis Agent

- 服务：`backend/financial_analysis_agent.py`
- 默认 off-path（解释/审计，不直接下单）
- 输出严格 schema（风险标记、驱动、阈值建议、解释）
- 可选 overlay 只允许降风险（例如 `pos_scale<=1`）

---

## 11. 一键链路命令

### 11.1 严格 e2e 验收

```bash
bash scripts/run_strict_e2e_acceptance.sh
```

### 11.2 Top50 升级流水线

```bash
bash scripts/run_top50_upgrade_pipeline.sh
```

### 11.3 Top50 训练短烟测

```bash
bash scripts/smoke_train_top50.sh
```

### 11.4 三主干成员训练

```bash
bash scripts/train_top50_ensemble_members.sh
```

---

## 12. 工件契约（strict）

`mlops_artifacts/validate.py` 要求工件至少包含：
- `manifest.json`
- `weights.pt`
- `schema_snapshot.yaml`
- `training_report.json`

缺任一文件直接 fail-fast，推理服务拒绝启动。

---

## 13. 关键环境变量

训练/模型：
- `LIQUID_BACKBONE` (`patchtst|itransformer|tft`)
- `LIQUID_D_MODEL`, `LIQUID_N_LAYERS`, `LIQUID_N_HEADS`, `LIQUID_PATCH_LEN`
- `LIQUID_LOOKBACK`（当前主线默认 2016）
- `LIQUID_SAMPLE_STRIDE_BUCKETS`
- `LIQUID_COST_PROFILE`

数据/审计：
- `DATABASE_URL`
- `LIQUID_UNIVERSE_TRACK`
- `LIQUID_DATA_READINESS_FILE`
- `AUDIT_MIN_MARKET_RATIO`
- `AUDIT_MIN_FEATURE_MATRIX_RATIO`
- `AUDIT_MIN_TEXT_COVERAGE`

执行/风控：
- `ENABLE_ACCOUNT_STATE_GUARD`
- `ENABLE_RISK_HARD_LIMITS`
- `ENABLE_STYLE_SWITCHING`
- `ENABLE_RECONCILIATION`

---

## 14. 测试与回归

```bash
pytest -q
```

重点测试覆盖：
- schema/contract 一致性
- cache roundtrip 与 as-of 无泄漏
- cost profile 训练/推理一致性
- model 输出 schema
- router 不塌缩
- agent schema/overlay 边界
- 执行守护（stoploss/ws降级/kill-switch）

---

## 15. 运行状态与进度查看

- 当前 ingest/merge 进度文档：`docs/INGEST_PROGRESS.md`
- 后台任务规范：`AGENTS.md`
- 长任务建议：
  - 优先使用稳定后台模式（`scripts/run_bg_task.sh`）
  - 或使用持久会话方式执行并定期轮询状态

---

## 16. 相关文档

- 升级总设计：`docs/upgrade_plan.md`
- 数据审计：`docs/OFFLINE_DATA_READINESS.md`
- 严格流水线：`docs/STRICT_PIPELINE_ZH.md`
- 部署指南：`docs/DEPLOYMENT_QUICKSTART_ZH.md`
- 趋势交易假设：`docs/TREND_TRADING_ASSUMPTIONS.md`（若存在）

