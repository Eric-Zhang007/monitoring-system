# 全网信息监测系统 - V2 双轨升级版（VC + Liquid）

**完成日期：** 2026-02-14
**项目状态：** ⚠️ 已完成 V2 核心重构，但 `liquid` 在严格门禁（`Sharpe_daily >= 1.5`）下尚未达标（暂不进入 2×A100 切换）

---

## 当前门禁快照（2026-02-15 14:18 UTC）
- 数据来源（strict, 生产口径过滤）：
  - `python3 scripts/evaluate_hard_metrics.py --track liquid --lookback-days 180 --score-source model --include-sources prod --exclude-sources smoke,async_test,maintenance --data-regimes prod_live`
  - `python3 scripts/check_backtest_paper_parity.py --track liquid --max-deviation 0.10 --min-completed-runs 5 --score-source model --include-sources prod --exclude-sources smoke,async_test,maintenance --data-regimes prod_live`
  - `python3 scripts/check_gpu_cutover_readiness.py`
- 当前值：
  - `Sharpe=0.45629`（未通过阈值 `>=1.5`）
  - `MaxDD=0.000178`（通过阈值 `<0.12`）
  - `execution_reject_rate=0.00244`（通过阈值 `<1%`）
  - `hard_passed=false`
  - `parity_30d=passed`（`relative_deviation=0.017046`）
  - `ready_for_gpu_cutover=false`（blocker: `hard_metrics_passed`）
- 结论：
  - 当前不应启动 2×A100 生产切换，仅建议继续 `paper + maintenance/prod_live` 校准迭代。

---

## V2 升级摘要（本次实现）

### 2026-02-15 15:10 UTC 执行层风控规则化 + 服务器离线部署脚本（本轮）
- 执行层风控新增硬规则（`backend/v2_router.py` + `backend/schemas_v2.py`）：
  - `RISK_SINGLE_STOP_LOSS_PCT`（默认 `0.018`）
  - `RISK_SINGLE_TAKE_PROFIT_PCT`（默认 `0.036`）
  - `RISK_INTRADAY_DRAWDOWN_HALT_PCT`（默认 `0.05`）
- `POST /api/v2/risk/check` 新增输入：
  - `latest_trade_edge_ratio`
  - `intraday_drawdown`
- `POST /api/v2/execution/run` 现在在执行前会进行：
  - 策略级连亏检查（既有）
  - 单笔止损触发检查（新增）
  - 单笔止盈触发检查（新增）
  - 日内回撤熔断检查（新增）
- `GET /api/v2/risk/limits` 返回项新增：
  - `max_daily_loss`
  - `max_consecutive_losses`
  - `single_trade_stop_loss_pct`
  - `single_trade_take_profit_pct`
  - `intraday_drawdown_halt_pct`
- 新增服务器离线部署脚本：
  - `scripts/server_preflight.sh`
  - `scripts/server_package_images.sh`
  - `scripts/server_upload_bundle.sh`
  - `scripts/server_seed_db.sh`
  - `scripts/server_bootstrap.sh`
  - `scripts/server_verify_runtime.sh`
- 服务器准备详细步骤见：`SERVER_PREP_PLAN_ZH.md`。

### 2026-02-15 14:18 UTC 并发重测与门禁纠偏（本轮）
- 并发执行链路修复：
  - `docker-compose.yml` 中 backend 改为 `uvicorn --workers ${BACKEND_UVICORN_WORKERS:-8}`。
  - 新增 `scripts/restart_backend_high_cpu.sh`，可按 worker 数重启 backend 压满 CPU。
  - `scripts/tune_liquid_strategy_grid.py` 新增并发与重试参数：
    - `--parallelism`
    - `--max-retries`
    - `--retry-backoff-sec`
- 一键验证脚本升级：
  - `scripts/run_2025_2026_validation_bundle.sh` 改为 `perp/spot` 回测并行 + `perp/spot` 调参并行；
  - 统一使用 `MIN_SHARPE_DAILY` 参数（默认 `1.5`）。
- readiness 门禁纠偏：
  - `scripts/check_gpu_cutover_readiness.py` 默认 `GPU_CUTOVER_MIN_SHARPE_DAILY` 上调至 `1.5`，避免低阈值误判绿灯。
- 2025 全年结果（Bitget 实盘历史）：
  - `perp`: `sharpe=-1.659584`, `pnl_after_cost=-0.022665`
  - `spot`: `sharpe=-4.682518`, `pnl_after_cost=-0.040771`
- 2025 至今严格门禁结果：
  - `no_leakage=passed`
  - `hard_metrics_420d=failed`（`sharpe_daily=-2.154071`）
  - `gpu_cutover_readiness_180d=false`（严格阈值下 blocker 为 `hard_metrics_passed`）

### 2026-02-15 P0 稳定化（本轮完善）
- 默认资产域收敛为加密中频：`LIQUID_SYMBOLS` 默认改为 `BTC,ETH,SOL`（`docker-compose` / `inference` / `training` / `backtest` 默认目标）。
- 风控违规码统一：`risk/check` 在 kill switch 命中时统一返回 `kill_switch_triggered:{track}:{strategy_id}`。
- 漂移统计修正：执行滑点样本仅统计 `filled|partially_filled`，不再纳入 `submitted`。
- 血缘一致性修正：`lineage/check` 在不指定 `target` 时按 `target` 分组比较最近两条快照，避免跨标的误比对。
- 回测口径修正：去除周末过滤，按加密 7x24 全时段计算 walk-forward 指标。
- WebSocket 入口收敛：前端默认连接 `/stream/signals`，不再默认连接被冻结的 `/ws`。
- 清理后端历史噪声：移除 `/ws` 冻结返回后的不可达旧逻辑，减少维护歧义。

### 2026-02-15 Phase-2 研究与生产闭环（本轮新增）
- 数据/特征闭环：
  - 训练与推理统一写入 `feature_snapshots`（含 `as_of_ts,event_time,data_version,lineage_id,target,track`）。
  - 训练新增 `train_lineage_id`，推理新增 `infer_lineage_id`，并写入预测关联字段。
  - `lineage/check` 支持严格模式和 `data_version`，返回 `mismatch_keys` 摘要。
  - 训练前数据质量门改为硬阻断：命中阈值返回 `blocked_by_data_quality`。
- 模型驱动回测：
  - `/api/v2/backtest/run` 主路径升级为“历史特征快照 + 指定模型版本”回放。
  - 返回补齐 `model_name/model_version/lineage_coverage/cost_breakdown`。
  - 失败原因标准化为 `model_artifact_missing|insufficient_features|insufficient_prices`。
- 执行与风控联动：
  - `risk/check` 新增 `daily_loss_exceeded`、`consecutive_loss_exceeded` 硬拦截。
  - `execution/run` 强制执行前风险校验，风险未通过返回 `423`。
  - 执行元数据保留统一 lifecycle 事件数组。
- 治理阈值化与审计：
  - `model_ops_scheduler` 全部阈值由 ENV 驱动，支持 drift/gate/rollback/rollout 统一参数化。
  - rollback 采用“连续失败窗口”触发，并返回 `windows_failed`、`trigger_rule`。
  - 调度器输出结构化审计日志（`who=system, source=scheduler, window, thresholds, decision`）。
- SLO 与告警：
  - 新增指标：`signal_latency`、`execution_latency`、`execution_reject_rate`、`data_freshness_seconds` 等。
  - `monitoring/health_check.py` 新增 SLO 判定：
    - `signal p95 < 150ms`
    - `execution p95 < 300ms`
    - 无样本时标记 `insufficient_observation`。
  - Prometheus 增加 P1/P2 规则与 `route` 标签（`monitoring/alerts.yml`）。
- 测试与烟测：
  - 新增测试：`test_execution_engine_paths.py`、`test_model_ops_decisions.py`、`test_lineage_replay_consistency.py`。
  - 扩展核心测试覆盖 runtime 风控违规、lineage mismatch keys。
  - `scripts/test_v2_api.sh` 增加关键字段/类型断言。

### 2026-02-15 Phase-4/5 稳定性与治理闭环（本轮新增）
- 调度治理闭环：
  - 新增 `GET /api/v2/models/rollout/state`，提供当前 rollout 阶段状态查询。
  - 新增 `POST /api/v2/models/audit/log`，用于 scheduler 审计持久化。
  - `model_ops_scheduler` rollout 由固定值改为动态读取当前阶段，按 `10 -> 30 -> 100` 阶梯推进；已达 `100` 时跳过推进并记录原因。
- 审计持久化：
  - 调度器每次 drift/gate/rollback/rollout 的审计尝试写入后端。
  - 后端通过 `risk_events` 统一存储 `scheduler_audit_log`，保留 `who/source/window/thresholds/decision`。
- WebSocket 可靠性：
  - 连接管理升级为“每连接独立队列 + sender task”，支持背压控制与慢连接隔离。
  - 支持批量 flush、发送超时、队列溢出自动摘除，避免广播阻塞。
  - 新增指标：`WEBSOCKET_DROPPED_MESSAGES_TOTAL{reason}`。
- 回归验收：
  - 容器内核心测试：`28 passed`。
  - `scripts/test_v2_api.sh` 全通过。

### 2026-02-15 Phase-6 指标达标优化（本轮新增）
- 硬指标口径与门禁分轨：
  - `scripts/evaluate_hard_metrics.py` 仅以 `completed` 回测样本统计收益指标。
  - 新增输出：`track_mode`、`failed_runs_count`、`failed_ratio`、`artifact_failure_ratio`。
  - `liquid` 执行硬门禁（`--enforce` 可阻断），`vc` 仅监控（`monitor_only=true`）。
- 执行拒单治理：
  - Paper 执行默认关闭随机拒单（`PAPER_ENABLE_RANDOM_REJECT=0`）。
  - 拒单改为可解释原因分类并回传 `reject_reason_category`。
  - `POST /api/v2/execution/run` 响应新增 `reject_breakdown` 聚合。
- 策略层强化：
  - 增加非线性仓位映射（score-to-size）、波动分层仓位上限、成本惩罚项。
  - 命中 drawdown 风险时自动缩减仓位上限，优先软降杠杆而非直接停机。
- 偏差治理：
  - 新增 `POST /api/v2/models/parity/check`（`passed|failed|insufficient_observation` 三态）。
  - `scripts/check_backtest_paper_parity.py` 支持 `7d/30d` 双窗口与 `min_completed_runs` 门槛。
  - 调度器已接入 parity 检查并写入治理审计。
- 新增指标与告警：
  - `ms_execution_rejects_total{adapter,reason}`
  - `ms_backtest_failed_runs_total{track,reason}`
  - `ms_metric_gate_status{track,metric}`
  - 告警新增 `ExecutionRejectRateCritical`（P1）与 `ExecutionRejectReasonSkew`（P2）。
- 交易所扩展：
  - 执行 adapter 新增 `bitget_live`，支持 `spot` 与 `perp_usdt` 参数透传。
  - `POST /api/v2/execution/run` 与 `POST /api/v2/execution/orders` 新增可选字段：`market_type/product_type/leverage/reduce_only/position_mode/margin_mode`。
  - 新增校验脚本：`scripts/validate_bitget_live.py`。

### 2026-02-15 Phase-6.3 指标治理口径收敛（本轮新增）
- 回测失败 supersede 治理：
  - `backtest_runs` 新增 supersede 字段（`superseded_by_run_id/supersede_reason/superseded_at`）。
  - 重放成功后可将历史 `model_artifact_missing` 失败样本标记 superseded，保留审计但不污染有效失败统计。
- hard metrics 统计升级：
  - `scripts/evaluate_hard_metrics.py` 默认使用“有效失败口径”（排除 superseded）。
  - 输出新增：`failed_runs_effective_count`、`artifact_missing_effective_count`、`superseded_runs_count`。
- parity 口径升级：
  - 从全局 PnL 代理比值切换为“同窗口 + 同 target 交集 + 已成交订单（filled/partially_filled）”比较。
  - `POST /api/v2/models/parity/check` 返回新增：
    - `matched_targets_count`
    - `paper_filled_orders_count`
    - `comparison_basis`
    - `window_details`
- 回测结果补充：
  - `/api/v2/backtest/run` completed 指标新增 `metrics.per_target`，用于 parity 按 target 对齐。

### 2026-02-15 Codex Plan 剩余八项落地（本轮新增）
- 告警触达闭环：
  - 新增 `alertmanager` 服务（`docker-compose`）与配置 `monitoring/alertmanager.yml`。
  - P1 告警按 5 分钟重复触发，P2 为 15 分钟。
  - 后端新增 `POST /api/v2/alerts/notify`，将告警写入 `risk_events` 审计表。
- SLO 扩展：
  - `monitoring/health_check.py` 新增 `p50/p95/p99` 与 API 可用性（目标 `>=99.9%`）判定。
  - 告警新增 `ApiAvailabilityLow` 与 `SignalLatencyP99Degraded`。
- 独立任务队列：
  - 新增 `task_worker` 容器（Redis 队列消费）。
  - 新增异步任务 API：
    - `POST /api/v2/tasks/backtest`
    - `POST /api/v2/tasks/pnl-attribution`
    - `GET /api/v2/tasks/{task_id}`
- 自动验收/治理脚本：
  - `scripts/check_backtest_paper_parity.py`（回测-paper 偏差）
  - `scripts/evaluate_hard_metrics.py`（Sharpe/MaxDD/拒绝率门槛）
  - `scripts/replay_model_run.py`（一键回放复现）
  - `scripts/validate_coinbase_live.py`（live 连通性验收）
  - `scripts/chaos_drill.py`（混沌演练）

### 2026-02-15 Phase1 加密中频基础落地（新增）
- 冻结旧口径 API：`/api/predictions*`、`/api/prices*`、`/api/news*`、`/api/indicators*`、`/ws` 返回 `410`，统一到 `/api/v2/*`。
- 新增执行 API：
  - `POST /api/v2/execution/orders`（提交订单）
  - `GET /api/v2/execution/orders/{order_id}`（查询订单）
  - `POST /api/v2/execution/run`（统一执行入口，支持 `paper|coinbase_live|bitget_live` + `time_in_force|max_slippage_bps|venue|market_type`）
  - `GET /api/v2/execution/audit/{decision_id}`（交易审计链路：`signal->order->fill->position->pnl`）
- 新增模型治理与监控 API：
  - `POST /api/v2/models/drift/evaluate`
  - `POST /api/v2/models/gate/auto-evaluate`
  - `POST /api/v2/models/rollout/advance`
  - `GET /api/v2/metrics/pnl-attribution`
  - `POST /api/v2/data-quality/lineage/check`
  - `GET /api/v2/risk/kill-switch`
  - `POST /api/v2/risk/kill-switch/trigger`
  - `POST /api/v2/risk/kill-switch/reset`
  - `GET /api/v2/data-quality/consistency`
- 新增可观测性：
  - `GET /metrics`（Prometheus 指标）
  - 指标覆盖：HTTP延迟/失败率、执行结果、漂移动作、风险硬拦截、WebSocket连接数
- 新增 Alembic 迁移：`20260215_0005_crypto_phase1_foundation.py`
  - 加密数据表：`market_bars`、`orderbook_l2`、`trades_ticks`、`funding_rates`、`onchain_signals`
  - `feature_snapshots` 增强字段：`as_of_ts`、`event_time`、`data_version`、`lineage_id`
  - `orders_sim` 扩展字段：`adapter`、`venue`、`time_in_force`、`max_slippage_bps`、`strategy_id`
- 训练链路增强：`TSMixer + LightGBM` 集成、数据质量 gate、固定随机种子、配置固化、early stopping、lr scheduler、checkpoint resume、OOM/NaN 降级重试、特征标准化参数持久化、`purged K-fold + walk-forward` 指标输出（IC/HitRate/PnL after cost/Turnover/MaxDD）。
- 推理链路增强：批量拉取价格/事件上下文 + 盘口/资金费率/链上信号，按 15 维特征推理，并与 `TSMixer + LightGBM` 集成路径对齐。

### 当前策略与参数口径（2026-02-15）
- 回测主路径：`_run_model_replay_backtest`（`backend/v2_router.py`）采用“特征打分 -> 仓位映射 -> 成本惩罚 -> 风险限幅”。
- 关键参数（backend ENV）：
  - `SIGNAL_ENTRY_Z_MIN`
  - `SIGNAL_EXIT_Z_MIN`
  - `POSITION_MAX_WEIGHT_BASE`
  - `POSITION_MAX_WEIGHT_HIGH_VOL_MULT`
  - `COST_PENALTY_LAMBDA`
  - `COST_FEE_BPS`
  - `COST_SLIPPAGE_BPS`
  - `COST_IMPACT_COEFF`
  - `BACKTEST_MAX_ABS_POSITION`
  - `BACKTEST_MAX_STEP_TURNOVER`
  - `BACKTEST_COST_EDGE_MULT`
  - `BACKTEST_ENTRY_Z_MIN`
  - `BACKTEST_EXIT_Z_MIN`
- 当前默认值以 `docker-compose.yml` 为准。
- 说明：文档下方早期“MVP说明”章节保留为历史记录，若与本节冲突，以本节与代码实现为准。

- 新增 `backend /api/v2/*`：
  - `POST /api/v2/ingest/events`
  - `GET /api/v2/entities/{entity_id}`
  - `POST /api/v2/predict/vc`
  - `POST /api/v2/predict/liquid`
  - `POST /api/v2/portfolio/score`
  - `GET /api/v2/predictions/{id}/explanation`
  - `POST /api/v2/backtest/run`
  - `GET /api/v2/backtest/{run_id}`
  - `POST /api/v2/signals/generate`
  - `POST /api/v2/portfolio/rebalance`
  - `GET /api/v2/risk/limits`
  - `POST /api/v2/risk/check`
  - `GET /api/v2/risk/kill-switch`
  - `POST /api/v2/risk/kill-switch/trigger`
  - `POST /api/v2/risk/kill-switch/reset`
  - `POST /api/v2/models/gate/evaluate`
  - `POST /api/v2/models/rollout/advance`
  - `POST /api/v2/models/rollback/check`
  - `POST /api/v2/execution/run`
  - `GET /api/v2/execution/audit/{decision_id}`
  - `POST /api/v2/data-quality/sample`
  - `POST /api/v2/data-quality/audit`
  - `GET /api/v2/data-quality/stats`
  - `GET /api/v2/data-quality/consistency`
- 新增 WebSocket 主题：
  - `/stream/events`
  - `/stream/signals`
  - `/stream/risk`
- 新增 Canonical Schema 与审计表：
  - `entities`, `events`, `event_links`, `feature_snapshots`
  - `model_registry`, `predictions_v2`, `prediction_explanations`, `backtest_runs`
  - `signal_candidates`, `orders_sim`, `positions_snapshots`
  - `risk_events`, `model_promotions`, `data_quality_audit`
- 采集器升级为插件化连接器：
  - `GDELT`, `RSS`, `SEC EDGAR`（真实信源接入路径）
- 训练与推理改为双轨模块：
  - `training/feature_pipeline.py`
  - `training/vc_model_trainer.py`
  - `training/liquid_model_trainer.py`
  - `inference/model_router.py`
  - `inference/explainer.py`
- 数据库迁移升级为 Alembic：
  - `backend/alembic.ini`
  - `backend/alembic/env.py`
  - `backend/alembic/versions/20260214_0001_v2_canonical_schema.py`
  - `backend/alembic/versions/20260215_0002_eval_execution_risk.py`
  - `backend/alembic/versions/20260215_0003_model_state_and_gate.py`
  - `backend/alembic/versions/20260215_0004_data_quality_review_fields.py`
  - `backend/alembic/versions/20260215_0006_risk_control_state.py`

---

## 📋 项目概述

> 说明：以下“项目结构/技术栈/成本”等章节包含历史 MVP 描述，用于架构参考；上线准入请以本文顶部“当前门禁快照”与 `TRACKING.md` 为准。

这是一个完整的金融信息监测与AI预测系统MVP，包含数据采集、实时监控、GPU加速推理、NLP情感分析和多响应式前端。

### 核心功能
- 📊 **实时价格监控**：支持多资产价格实时追踪
- 📰 **新闻情感分析**：NIM NLP 模型提取语义特征
- 🎯 **AI 价格预测**：LSTM/GRU 模型预测 1h/1d/7d 趋势
- 📱 **响应式前端**：桌面/平板/移动端全适配
- 🎨 **颜色方案切换**：支持国内（红涨绿跌）和国际（绿涨红跌）标准

---

## 📂 项目结构

```
monitoring-system/
├── docker-compose.yml              ✅ Docker Compose 编排
├── nginx/nginx.conf                ✅ 反向代理 + WebSocket + 限流
│
├── backend/                        ✅ FastAPI 后端服务
│   ├── main.py                     - FastAPI 主程序
│   ├── gpu_manager.py              - GPU 资源管理器
│   ├── nim_integration.py          - NIM 特征缓存（SQL注入已修复）
│   ├── redis_streams.py            - Redis Streams 生产者/消费者（XACK确认）
│   ├── Dockerfile
│   └── requirements.txt
│
├── collector/                      ✅ 数据采集器
│   ├── collector.py                - 新闻/价格数据采集
│   ├── Dockerfile
│   └── requirements.txt
│
├── inference/                      ✅ 推理服务（GPU 0 - 实时）
│   ├── main.py                     - PyTorch 推理服务
│   ├── Dockerfile
│   └── requirements.txt
│
├── training/                       ✅ 训练服务（GPU 1 - 离线任务）
│   ├── main.py                     - 模型训练 + NIM特征提取
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                       ✅ React + TypeScript + Tailwind
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── index.html
│   ├── src/
│   │   ├── App.tsx                 - 主应用（响应式导航）
│   │   ├── main.tsx
│   │   ├── types.ts                - TypeScript 类型定义
│   │   ├── contexts/
│   │   │   └── ColorSchemeContext.tsx  - 颜色方案切换
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts     - WebSocket 连接
│   │   └── components/
│   │       ├── Header.tsx          - 头部导航
│   │       ├── Hero.tsx            - 横幅
│   │       ├── Dashboard.tsx       - 仪表盘
│   │       ├── NewsSection.tsx     - 新闻列表
│   │       ├── PredictionsSection.tsx  - 预测卡片
│   │       ├── MonitorPanel.tsx    - 系统监控
│   │       └── MobileNav.tsx       - 移动端底部导航
│   └── Dockerfile
│
├── monitoring/                     ✅ 监控脚本
│   └── system_monitor.py           - GPU/内存/服务健康监控
│
└── scripts/                        ✅ 部署脚本
    ├── test.sh                     - 测试脚本（已通过✅）
    └── deploy.sh                   - 一键部署脚本
```

---

## 🔑 关键修复与亮点

### 1. GPU 资源管理（修复版）
```
GPU 0: 实时推理（LSTM/GRU 1h/1d 预测）- 24h 序列
GPU 1: 训练 + NIM 离线特征提取 - 30d 序列
```
- ✅ 避免 NIM 批量任务影响实时推理
- ✅ 系统内存监控（新功能）

### 2. SQL 注入安全防护
**修复前：**
```python
cur.execute(f"... timestamp > NOW() - INTERVAL '{max_age_hours}' hours")
```
**修复后：**
```python
cur.execute("""... timestamp > NOW() - make_interval(secs => %s * 3600)""",
           (max_age_hours,))
```

### 3. Redis Streams 消息确认
```python
# 处理消息后确认
self.redis.xack(stream_name, self.consumer_group, message_id)
```

### 4. Docker 配置完善
- ✅ 所有服务：`restart: unless-stopped`
- ✅ 资源限制：CPU/内存/GPU 约束
- ✅ 健康检查：HEALTHCHECK 指令

### 5. 前端响应式设计
- ✅ 大屏（>1600px）：三栏布局
- ✅ 中屏（1200-1600px）：两栏布局
- ✅ 小屏（<1200px）：单栏 + Tab 导航
- ✅ 移动端（<768px）：底部导航

### 6. 颜色方案切换
- 🇨🇳 中国：红涨绿跌
- 🇺🇸 国际：绿涨红跌
- 实时切换，无刷新

---

## ✅ 测试结果

### 语法检查
```
✅ 所有 Python 文件编译通过
✅ 前端 TypeScript 编译通过（0 错误）
✅ Docker Compose 配置有效
```

### 前端构建
```
✓ built in 4.32s
dist/index.html                   0.51 kB │ gzip:  0.37 kB
dist/assets/index.css             0.56 kB │ gzip:  0.28 kB
dist/assets/index.js            181.76 kB │ gzip: 54.96 kB
```

### 文件完整性
```
✅ 所有必需文件已创建
✅ Dockerfile 配置正确
✅ requirements.txt 完整
```

---

## 🚀 快速开始

### 1. 一键部署
```bash
cd /home/admin/.openclaw/workspace/monitoring-system
./scripts/deploy.sh
```

### 2. 访问系统
- **前端界面**：http://localhost
- **后端 API**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs
- **Prometheus**：http://localhost:9090
- **Grafana 监控**：http://localhost:3000 (admin/admin)

### 3. 常用命令
```bash
# 查看服务状态
docker compose ps

# 执行数据库迁移
docker compose run --rm orchestrator

# 查看日志
docker compose logs -f backend

# 停止服务
docker compose down

# 重启服务
docker compose restart [service_name]

# 运行扩展版 V2 API 冒烟测试（29项）
API_BASE=http://localhost:8000 ./scripts/test_v2_api.sh

# 异步任务队列状态查看（示例）
curl -s "http://localhost:8000/api/v2/tasks/<task_id>"

# 量化硬指标统计（Sharpe/MaxDD/拒绝率）
python3 scripts/evaluate_hard_metrics.py --track liquid

# 一键回放最近一次 backtest 配置并比对差异
python3 scripts/replay_model_run.py --tolerance 1e-6

# 回测-paper 偏差自动验收
python3 scripts/check_backtest_paper_parity.py --track liquid --max-deviation 0.10 --min-completed-runs 5

# Phase4/5 告警规则阈值验收
python3 scripts/validate_phase45_alerts.py

# CI 门禁（hard_metrics + parity + alerts；不通过返回非 0）
bash scripts/ci_phase45_gate.sh

# 主动触发 parity API 检查（供调度/人工验证）
curl -s -X POST http://localhost:8000/api/v2/models/parity/check \
  -H "content-type: application/json" \
  -d '{"track":"liquid","max_deviation":0.10,"min_completed_runs":5}'

# Coinbase live 连通性验收（无密钥会返回 skipped）
python3 scripts/validate_coinbase_live.py

# Bitget live 连通性验收（无密钥会返回 skipped）
python3 scripts/validate_bitget_live.py

# 混沌演练（示例：中断 Redis，再 recover）
python3 scripts/chaos_drill.py redis_interrupt
python3 scripts/chaos_drill.py recover

# 回放失败 liquid 回测，补齐 completed 样本
python3 scripts/rebuild_liquid_completed_backtests.py --limit 30

# 严格口径批量回测（prod+model+prod_live）
python3 scripts/run_prod_live_backtest_batch.py --api-base http://localhost:8000 --n-runs 12 --fee-bps 0.5 --slippage-bps 0.2 --signal-entry-z-min 0.08 --signal-exit-z-min 0.028 --position-max-weight-base 0.08 --cost-penalty-lambda 1.0 --signal-polarity-mode auto_train_ic

# 清理旧 completed 样本（仅保留最近N条参与 hard gate 统计）
python3 scripts/supersede_stale_backtests.py --track liquid --keep-latest 20

# 校验回测 metrics contract（缺字段样本直接标红）
python3 scripts/validate_backtest_contracts.py --track liquid --lookback-days 180 --enforce

# 分析 backtest vs paper 的 target 偏差来源（含 fee/slippage/impact 成本归因）
python3 scripts/analyze_parity_gap.py --track liquid --window-days 30 --score-source model --include-sources prod --exclude-sources smoke,async_test,maintenance --data-regimes prod_live

# 策略参数网格调优（含反零交易约束）
python3 scripts/tune_liquid_strategy_grid.py --run-source prod --data-regime prod_live --score-source model --max-trials 64 --min-turnover 0.05 --min-trades 5 --min-abs-pnl 1e-5 --min-active-targets 2

# Phase-6.3 每日维护（重放+门禁+日报）
bash scripts/daily_phase63_maintenance.sh

# 持续修正循环（审查->选参->测试->门禁）
python3 scripts/continuous_remediation_loop.py --api-base http://localhost:8000 --max-iterations 0 --green-windows 3 --candidate-source auto --candidate-top-k 8 --candidate-refresh-every 3 --fee-bps 0.5 --slippage-bps 0.2 --signal-polarity-mode auto_train_ic

# 检查是否满足 GPU 切换门禁
python3 scripts/check_gpu_cutover_readiness.py

# 如需启动 GPU 推理/训练服务（默认 compose up 不启动这两个服务）
docker compose --profile gpu up -d inference training

# 本地一键安装依赖并运行 backend 单元测试
./scripts/dev_test.sh

# 每周数据质量抽样（默认200条）并导出审计清单
python3 scripts/data_quality_weekly_audit.py --api-base http://localhost:8000 --limit 200
```

---

## 📊 技术栈

### 后端
- FastAPI 0.104+ - 高性能异步框架
- Redis Streams - 替代 Kafka（MVP简化）
- PostgreSQL 16 + PGVector - 数据库 + 向量存储
- ClickHouse - 时序数据存储
- PyTorch 2.x - 深度学习框架

### 前端
- React 18 + TypeScript 5
- Vite - 构建工具
- Tailwind CSS 3 - 响应式样式
- Recharts - 图表库
- Lucide React - 图标库

### DevOps
- Docker Compose - 容器编排
- Nginx - 反向代理
- Grafana - 监控可视化

---

## 💰 成本估算（月）

| 配置项 | 成本 |
|--------|------|
| 笔记本本地训练/测试 | ¥0（算力成本按 0 计） |
| 2×A100 GPU（AutoDL 按时） | ¥11.96/小时（包天/包月通常更低） |
| 应用服务器（4 vCPU, 8GB） | ¥200 |
| PostgreSQL + PGVector | ¥150 |
| Redis | ¥50 |
| ClickHouse | ¥150 |
| Grafana + 监控 | ¥200 |
| **总计** | **按 GPU 使用时长线性增长** |

### HPO 分阶段算力成本（建议口径）

| 阶段 | 目标 | 推荐算力 | 默认时长 | 估算方式 |
|------|------|----------|----------|----------|
| Stage 1 | 粗搜（单交易对、短窗口） | `local` | 1.5h | `cpu_hourly_cny * cpu_units * hours` |
| Stage 2 | 候选精修（多交易对、中窗口） | `local` | 4h | `cpu_hourly_cny * cpu_units * hours` |
| Stage 3 | 前瞻 OOS（仅 prod） | `a100x2` | 8h | `2 * a100_hourly_cny * hours + cpu_overhead` |

可通过 HPO 脚本直接输出阶段成本估算：

```bash
python3 scripts/optuna_liquid_hpo.py \
  --stage 3 \
  --compute-tier a100x2 \
  --n-trials 80 \
  --a100-hourly-cny 11.96 \
  --cpu-hourly-cny 0 \
  --billing-mode hourly
```

---

## 🎓 认可说明

本系统由以下团队协作完成：

- **小黑（我）**：总体架构设计、项目协调、MVP 开发、测试验证
- **小蓝**：架构审查（5 must-fix + 5 should-fix 问题识别）
- **小黄**：前端设计审查（UI/UX 改进、响应式方案）

---

## 📝 后续优化建议

### Phase 2 功能
1. 添加 ClickHouse 数据存储
2. 实现 NIM 实时 API 调用集成
3. 添加历史回测功能
4. 实现多股票对比
5. 添加 Telegram 通知推送

### 性能优化
1. 模型量化（INT8）
2. 批量推理优化
3. Redis 缓存策略
4. CDN 加速前端

---

**报告生成时间：** 2026-02-15（持续更新）
**状态：** 主链路可用但 `liquid` 严格门禁未达标；当前建议仅 `paper` 运行，待 `Sharpe_daily >= 1.5` 连续窗口转绿后再灰度实盘。

<!-- AUTO_STATUS_SNAPSHOT:BEGIN -->
### Auto Snapshot (2026-02-15 14:18 UTC)
- track: `liquid`
- score_source: `model`
- sharpe: `0.45629`
- max_drawdown: `0.000178`
- execution_reject_rate: `0.00244`
- hard_passed: `false`
- parity_status: `passed`
- parity_matched_targets: `3`
- parity_paper_filled_orders: `1373`
<!-- AUTO_STATUS_SNAPSHOT:END -->
