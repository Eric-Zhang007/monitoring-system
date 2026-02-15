# 全网信息监测系统 - V2 双轨升级版（VC + Liquid）

**完成日期：** 2026-02-14
**项目状态：** ✅ 已完成 V2 核心重构（保留 V1 兼容接口）

---

## V2 升级摘要（本次实现）

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
  - `POST /api/v2/execution/run`（统一执行入口，支持 `paper|coinbase_live` + `time_in_force|max_slippage_bps|venue`）
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
python3 scripts/check_backtest_paper_parity.py --track liquid --target BTC

# Coinbase live 连通性验收（无密钥会返回 skipped）
python3 scripts/validate_coinbase_live.py

# 混沌演练（示例：中断 Redis，再 recover）
python3 scripts/chaos_drill.py redis_interrupt
python3 scripts/chaos_drill.py recover

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
| 2×A100 GPU（AutoDL 按时计费） | ¥2000-5000 |
| 应用服务器（4 vCPU, 8GB） | ¥200 |
| PostgreSQL + PGVector | ¥150 |
| Redis | ¥50 |
| ClickHouse | ¥150 |
| Grafana + 监控 | ¥200 |
| **总计** | **¥2750-5750** |

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

**报告生成时间：** 2026-02-14 19:35
**状态：** MVP 已完成，可部署上线 ✅
