# 全网信息监测系统 - V2 双轨升级版（VC + Liquid）

**完成日期：** 2026-02-14
**项目状态：** ✅ 已完成 V2 核心重构（保留 V1 兼容接口）

---

## V2 升级摘要（本次实现）

### 2026-02-15 Phase1 加密中频基础落地（新增）
- 冻结旧口径 API：`/api/predictions*`、`/api/prices*`、`/api/news*`、`/api/indicators*`、`/ws` 返回 `410`，统一到 `/api/v2/*`。
- 新增执行 API：
  - `POST /api/v2/execution/orders`（提交订单）
  - `GET /api/v2/execution/orders/{order_id}`（查询订单）
  - `POST /api/v2/execution/run`（统一执行入口，支持 `paper|coinbase_live` + `time_in_force|max_slippage_bps|venue`）
- 新增模型治理与监控 API：
  - `POST /api/v2/models/drift/evaluate`
  - `POST /api/v2/models/gate/auto-evaluate`
  - `GET /api/v2/metrics/pnl-attribution`
- 新增 Alembic 迁移：`20260215_0005_crypto_phase1_foundation.py`
  - 加密数据表：`market_bars`、`orderbook_l2`、`trades_ticks`、`funding_rates`、`onchain_signals`
  - `feature_snapshots` 增强字段：`as_of_ts`、`event_time`、`data_version`、`lineage_id`
  - `orders_sim` 扩展字段：`adapter`、`venue`、`time_in_force`、`max_slippage_bps`、`strategy_id`
- 训练链路增强：数据质量 gate、固定随机种子、时序验证、early stopping、lr scheduler、checkpoint resume、OOM 降级重试、特征标准化参数持久化。
- 推理链路增强：批量拉取价格/事件上下文，按 8 维特征（含多窗口收益/波动）推理，兼容模型维度自动对齐。

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
  - `POST /api/v2/models/gate/evaluate`
  - `POST /api/v2/models/rollback/check`
  - `POST /api/v2/execution/run`
  - `POST /api/v2/data-quality/sample`
  - `POST /api/v2/data-quality/audit`
  - `GET /api/v2/data-quality/stats`
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

# 运行扩展版 V2 API 冒烟测试（17项）
API_BASE=http://localhost:8000 ./scripts/test_v2_api.sh

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
