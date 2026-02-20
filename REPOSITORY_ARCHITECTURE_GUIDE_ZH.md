# Monitoring System 仓库全景说明（面向架构升级与重构）

## 1. 文档目标

这份文档用于帮助新开发者在最短时间内掌握本仓库：

1. 系统在做什么、如何跑起来、核心链路在哪里。
2. 哪些代码是核心生产路径，哪些是历史兼容或辅助脚本。
3. 当前架构的主要风险点与重构切入顺序。
4. 如何在不破坏现有运行能力的前提下做架构升级。

本文基于当前代码仓库实际实现（`backend/`、`collector/`、`frontend/`、`monitoring/`、`training/`、`scripts/`）整理。

---

## 2. 系统定位与总体架构

### 2.1 系统定位

这是一个围绕“事件采集 -> 特征构建 -> 预测/信号 -> 风控 -> 模拟执行 -> 监控运维”的一体化量化监控系统，当前主轨道是 `liquid`（加密资产，如 `BTC/ETH/SOL`），同时保留 `vc` 轨道。

### 2.2 单机部署拓扑（当前默认）

根据 `README.md` 与 `scripts/server_up.sh`，默认会拉起：

1. `backend`（FastAPI，`8000`）
2. `collector`（多源采集）
3. `task_worker`（异步任务执行）
4. `model_ops`（模型运维调度）
5. `ops_loop`（可选，持续运行循环）

依赖组件：

1. PostgreSQL（主存储）
2. Redis（任务队列 + Streams）
3. 可选 GPU（训练/推理加速）

### 2.3 关键业务数据流（生产主链）

1. `collector/collector.py` 拉取外部数据源，归一化后调用 `POST /api/v2/ingest/events`。
2. `backend/v2_router.py` 通过 `V2Repository` 写入 `events/entities/event_links/data_quality_audit`。
3. 预测请求走 `POST /api/v2/predict/liquid`（`LiquidModelService` + `ModelRouter` + DB 最新特征/价格上下文）。
4. 信号生成走 `POST /api/v2/signals/generate`。
5. 执行链路走 `POST /api/v2/execution/orders` + `POST /api/v2/execution/run`，并经过风险校验与 kill-switch。
6. 结果落地 `orders_sim/risk_events/predictions_v2` 等表，监控面板读取 `/api/status`、`/api/v2/monitor/*`、`/api/v2/risk/*` 等接口。

---

## 3. 仓库结构速览（按职责分组）

### 3.1 在线服务层

1. `backend/main.py`
2. `backend/v2_router.py`
3. `backend/v2_repository.py`
4. `backend/execution_engine.py`
5. `backend/task_queue.py`

### 3.2 数据采集与增强

1. `collector/collector.py`
2. `collector/connectors/*`
3. `collector/entity_linking.py`
4. `collector/llm_enrichment.py`

### 3.3 推理与特征契约

1. `inference/model_router.py`
2. `inference/liquid_feature_contract.py`
3. `backend/liquid_model_service.py`
4. `inference/main.py`（更像离线/独立推理服务实现）

### 3.4 训练与验证

1. `training/feature_pipeline.py`
2. `training/liquid_model_trainer.py`
3. `training/vc_model_trainer.py`
4. `training/train_multimodal.py`
5. `training/eval_multimodal_oos.py`
6. `training/register_candidate_model.py`
7. `training/backbone_experiments.py`

### 3.5 运维与门禁

1. `scripts/server_preflight.sh`
2. `scripts/server_up.sh`
3. `scripts/server_readiness.sh`
4. `monitoring/model_ops_scheduler.py`
5. `monitoring/task_worker.py`
6. `monitoring/paper_trading_daemon.py`
7. `scripts/continuous_ops_loop.py`
8. `scripts/rehearse_server_release_rollback.sh`

### 3.6 前端

1. `frontend/src/App.tsx`
2. `frontend/src/components/MonitorPanel.tsx`
3. `frontend/src/components/V2Lab.tsx`
4. `frontend/src/components/Dashboard.tsx`

---

## 4. 核心模块深读

## 4.1 Backend（核心控制平面 + 业务 API）

### 4.1.1 `backend/main.py`

职责：

1. FastAPI 应用生命周期、CORS、全局异常处理。
2. 暴露 `/health`、`/metrics`、`/api/status`。
3. 提供 `/api/v2/monitor/*` 与 `/api/v2/control/*`（大量读写 `artifacts/*.json(.jsonl)`）。
4. 管理 WebSocket `/stream/events|signals|risk`。
5. `app.include_router(v2_router)` 挂载全部核心业务接口。

重要事实：

1. 旧版 `/api/*` 与 `/ws` 已冻结，统一返回 `410`（强制迁移到 `/api/v2/*`）。
2. 内置 Redis stream 消费与广播任务（`consume_redis_messages`、`broadcast_predictions`）。

### 4.1.2 `backend/v2_router.py`

这是业务中心文件（约 3600+ 行），覆盖：

1. 事件入库：`/api/v2/ingest/events`
2. 预测：`/api/v2/predict/vc`、`/api/v2/predict/liquid`
3. 回测与异步任务：`/api/v2/backtest/run`、`/api/v2/tasks/backtest`
4. 信号：`/api/v2/signals/generate`
5. 风控：`/api/v2/risk/*`
6. 执行：`/api/v2/execution/orders`、`/api/v2/execution/run`、`/api/v2/execution/audit/{decision_id}`
7. 模型运维：drift/gate/rollback/rollout/parity
8. 数据质量：sample/stats/consistency/lineage/audit

关键设计点：

1. 强风控前置：执行前会做波动、连续亏损、日内损失、kill-switch、仓位风险校验。
2. 运行参数高度 env 化：成本、门禁阈值、run_source/data_regime 过滤、rollout 阶梯等。
3. 指标埋点完整：信号延迟、执行延迟、拒单率、漂移事件等。

### 4.1.3 `backend/v2_repository.py`

职责：

1. 所有核心 DB 读写聚合入口（连接池 + SQL）。
2. 事件、预测、回测、执行、风控、模型状态、数据质量、血缘一致性等持久化。
3. 兼具 `query + business-ish assembly`（例如 PnL attribution、lineage consistency）。

现状：

1. 体量很大（约 2000+ 行），已开始拆分到 `backend/repository_modules/*`，但仍是单大类。

### 4.1.4 `backend/execution_engine.py`

执行适配器：

1. `PaperExecutionAdapter`（默认模拟执行）
2. `CoinbaseLiveAdapter`
3. `BitgetLiveAdapter`
4. `ExecutionEngine`（统一调度）

设计特点：

1. Paper 支持部分成交、超时重试、拒单注入（仿真）。
2. Live 适配器具备错误归一化，便于风控与指标统计。

### 4.1.5 `backend/task_queue.py` + `monitoring/task_worker.py`

模式：Redis list + result key。

1. `enqueue_task` 入队。
2. worker `claim_next_task` + `set_task_status`。
3. 支持 stale processing requeue。
4. 当前任务类型主要为 `backtest_run` 与 `pnl_attribution`。

## 4.2 Collector（采集平面）

`collector/collector.py` 是多源连接器编排器：

1. connectors 插件化（RSS/GDELT/SEC/FRED/CoinGecko/社媒）。
2. 采集失败重试 + cooldown（简易断路器）。
3. 支持 LLM enrichment（`collector/llm_enrichment.py`，OpenAI-compatible）。
4. 采集后调用后端 ingest API，而不是直接写 DB。
5. 内置 Prometheus metrics（collector 自身监控）。

## 4.3 推理与特征契约

### 4.3.1 线上主路径

当前线上预测主要在 backend 内完成：

1. `backend/liquid_model_service.py`
2. `inference/model_router.py`
3. `inference/liquid_feature_contract.py`

### 4.3.2 关键机制

1. 模型选择优先使用 `active_model_state`，否则默认模型。
2. 模型工件检查严格：缺失工件时返回 503 并阻断模型预测。
3. 特征 schema/data version 有兼容集，防止误载不兼容模型。

### 4.3.3 关于 `inference/main.py`

从 `scripts/server_up.sh` 未启动该服务这一事实推断：

1. `inference/main.py` 更像历史独立服务或离线推理/回填实现。
2. 当前实时 API 主路由已内嵌在 backend 的 v2 接口链中。

## 4.4 训练系统

### 4.4.1 特征与训练

1. `training/feature_pipeline.py`：从 DB 构建训练样本，做 DQ 检查，强调 as-of 防泄露。
2. `training/liquid_model_trainer.py`：TSMixer + LightGBM/线性兜底，多种验证切片（walk-forward/purged kfold）。
3. `training/vc_model_trainer.py`：VC 轨道模型训练。

### 4.4.2 候选注册

`training/register_candidate_model.py` 同时：

1. 追加 `artifacts/models/candidate_registry.jsonl`
2. 写入 `candidate_models` 表

## 4.5 监控与运维自动化

1. `monitoring/model_ops_scheduler.py` 周期触发 drift/gate/parity/rollout/rollback 检查。
2. 同一调度器已接入多模态 gate（读取 `/api/v2/monitor/model-status`，写入 `artifacts/ops/multimodal_gate_state.json`）。
3. `monitoring/paper_trading_daemon.py` 执行持续模拟盘闭环。
4. `scripts/continuous_ops_loop.py` 执行“paper + ingest/train/backtest/parity/audit”循环。
5. `scripts/server_preflight.sh` 与 `scripts/server_up.sh` 构成上线前后硬门禁。
6. `scripts/rehearse_server_release_rollback.sh` 用于发布/回滚全流程预演（默认 dry-run）。

## 4.6 前端

### 4.6.1 结构

1. React + Vite + TS + Tailwind。
2. `MonitorPanel` 深度接入真实 v2 监控与控制接口。
3. `V2Lab` 可直接发起 ingest/predict 调试请求。

### 4.6.2 注意点

`Dashboard.tsx` 仍使用本地 mock 数据，不是生产真实行情展示链路。

---

## 5. 数据库模型（按业务域）

Alembic 迁移位于 `backend/alembic/versions/`，主线从 `20260214_0001` 到 `20260215_0011`。

### 5.1 事件与实体域

1. `entities`
2. `events`
3. `event_links`
4. `data_quality_audit`
5. `data_quality_review_logs`

### 5.2 特征与模型域

1. `feature_snapshots`
2. `model_registry`
3. `active_model_state`
4. `model_promotions`
5. `model_rollout_state`
6. `asset_universe_snapshots`

### 5.3 预测与交易域

1. `predictions_v2`
2. `prediction_explanations`
3. `signal_candidates`
4. `orders_sim`
5. `positions_snapshots`
6. `risk_events`

### 5.4 回测与绩效域

1. `backtest_runs`（含 supersede/run_source 扩展）

### 5.5 行情与市场微结构域

1. `market_bars`
2. `orderbook_l2`
3. `trades_ticks`
4. `funding_rates`
5. `onchain_signals`

---

## 6. 配置与运行入口

### 6.1 环境变量入口

1. 示例文件：`.env.example`
2. 启动脚本统一 source：`scripts/server_up.sh`
3. 安全相关约束：`backend/security_config.py`（显式 CORS allowlist，禁止 `credentials + *`）

### 6.2 一键运行相关脚本

1. 环境准备：`scripts/bootstrap_env.sh`
2. 预检：`scripts/server_preflight.sh`
3. 启动：`scripts/server_up.sh`
4. 停止：`scripts/server_down.sh`
5. 就绪检查：`scripts/server_readiness.sh`

---

## 7. 测试与质量门禁现状

`backend/tests/` 下测试覆盖面较广，重点在：

1. API 核心逻辑（`test_v2_router_core.py`）
2. 仓储层一致性（`test_v2_repository_utils.py`）
3. 执行与风控（execution/risk/parity/gate）
4. 任务队列与 worker
5. 监控健康、WebSocket 背压、安全配置

上线脚本会执行针对性测试集（见 `scripts/server_preflight.sh`）。

---

## 8. 当前架构优点与主要问题

## 8.1 优点

1. API 面完整，业务闭环已形成（采集-预测-执行-监控）。
2. 风控策略较严格，具备 kill-switch 与多重阻断。
3. 观测性较好：Prometheus 指标 + 健康检查 + 审计日志。
4. 运行脚本化程度高，便于单机交付。

## 8.2 主要问题（重构优先）

1. `v2_router.py` 超大，协议层与业务决策/风控/回测耦合严重。
2. `v2_repository.py` 超大，查询、聚合、业务语义混在一起。
3. 特征键与 schema 常量在多个模块重复定义，存在漂移风险。
4. 在线状态同时使用 DB 与 `artifacts/*.json` 文件，状态源不统一。
5. 前端部分页面仍是 mock，认知上容易与真实交易链路混淆。
6. 存在历史路径并行（如 `inference/main.py`、旧表/兼容逻辑），边界不够清晰。

---

## 9. 面向升级与重构的建议路线

## 9.1 Phase A：先拆接口层（低风险高收益）

目标：把 `v2_router.py` 按业务域拆为多个 router。

建议分组：

1. `routers/ingest.py`
2. `routers/predict.py`
3. `routers/backtest.py`
4. `routers/risk.py`
5. `routers/execution.py`
6. `routers/model_ops.py`
7. `routers/data_quality.py`

收益：

1. 降低认知负担。
2. 更容易做并行开发与代码评审。

## 9.2 Phase B：分层抽象（Service + Repository）

目标：隔离“HTTP 协议层 / 业务层 / 数据访问层”。

建议：

1. router 只做参数校验与响应编排。
2. `services/*` 承担策略和流程（风控、回测门禁、执行前检查）。
3. repository 只保留 CRUD/查询，不承载复杂决策。

## 9.3 Phase C：统一特征契约

目标：唯一特征 schema 源。

建议：

1. 以 `inference/liquid_feature_contract.py` 为单一事实源。
2. 训练、推理、回测都只 import 该定义。
3. CI 加“契约漂移检测”（字段数、字段顺序、版本号）。

## 9.4 Phase D：状态一致性收敛

目标：减少 DB 与文件双写状态。

建议：

1. 将关键控制状态（live 开关、candidate 切换、ops 状态）逐步收敛入 DB。
2. `artifacts/*` 仅保留导出快照，不作为权威状态源。

## 9.5 Phase E：前端真实数据化

目标：降低“看起来在线，实际 mock”的误判。

建议：

1. 给 `Dashboard` 增加真实 API 模式开关，并默认真实。
2. mock 数据仅用于开发模式，并有明确 UI 标记。

---

## 10. 重构过程中的“不破坏”清单

1. 不改动已有 `/api/v2/*` 对外契约（先兼容，后渐进迁移）。
2. 保持 `scripts/server_up.sh` 可启动性不回退。
3. 所有 kill-switch 与 risk-block 语义必须回归测试。
4. 回测门禁（strict as-of、gate 规则）不得被弱化。
5. 迁移脚本只追加不覆写，保证可回滚。

---

## 11. 新开发者 30 分钟上手路径

1. 先读 `README.md`（部署/门禁/启动总览）。
2. 再读 `backend/main.py` + `backend/v2_router.py`（API 全貌）。
3. 再读 `backend/v2_repository.py`（数据模型与落库路径）。
4. 再读 `collector/collector.py`（数据输入路径）。
5. 最后读 `monitoring/model_ops_scheduler.py` + `monitoring/paper_trading_daemon.py`（持续运行逻辑）。

建议配合命令：

```bash
bash scripts/bootstrap_env.sh
bash scripts/server_preflight.sh
bash scripts/server_up.sh
bash scripts/server_readiness.sh
```

---

## 12. 结论

当前仓库已具备完整的端到端闭环与生产化脚本基础，最需要做的是“结构化拆分与状态收敛”，而不是重写业务逻辑。建议按“先拆 router -> 再拆 service/repository -> 再统一契约和状态”的顺序推进，这样能在保持可运行的前提下稳定完成架构升级。
