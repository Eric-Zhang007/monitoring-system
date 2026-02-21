# Monitoring System 服务器部署手册

本文件是项目唯一运行与部署说明，目标是让你在一台 Linux 服务器上可重复完成：初始化、数据门禁、服务启动、前端发布、持续运维。

架构与重构全景说明请见：`REPOSITORY_ARCHITECTURE_GUIDE_ZH.md`

## 1. 部署目标

服务拓扑（单机）：
- `backend`：FastAPI API 与监控接口（端口 `8000`）
- `collector`：多源采集
- `task_worker`：异步任务执行
- `model_ops`：模型运维调度
- `ops_loop`：持续运行状态循环（可选）
- `paper_trading_daemon`：持续模拟盘守护

人工控制原则：
- 不自动晋级实盘
- 仅通过 API/前端人工开关实盘

## 2. 服务器要求

基础要求：
- OS：Ubuntu 22.04+（或同类 Linux）
- Python：3.10+
- PostgreSQL：14+
- Redis：6+
- `screen`、`curl`、`git`
- Node.js 18+（前端构建）

资源建议：
- CPU：8 vCPU+
- 内存：16 GB+
- 磁盘：200 GB+
- GPU：可选（训练阶段建议）

网络端口：
- `8000`：后端 API
- `3001`：前端（如本地静态预览）
- `5432`：PostgreSQL（建议内网）
- `6379`：Redis（建议内网）

## 3. 首次部署

### 3.1 克隆与目录

```bash
git clone <your-repo-url> monitoring-system
cd monitoring-system
```

### 3.2 系统依赖安装（Ubuntu 示例）

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip postgresql redis-server screen curl git nodejs npm
```

### 3.3 环境变量

```bash
cp .env.example .env
```

必须至少确认：
- `DATABASE_URL`
- `REDIS_URL`
- `LIQUID_SYMBOLS`
- `LIQUID_PRIMARY_TIMEFRAME=5m`

### 3.4 Python 运行环境

```bash
bash scripts/bootstrap_env.sh
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
```

## 4. 上线前固定门禁（串行）

### 4.0 国内服务器离线策略（推荐）

如果服务器在国内且网络受限，默认采用：
1. 本地机器采集并构建离线数据包
2. 通过 `scp` 上传到服务器，或推送到 GitHub 后在服务器 `git clone/pull`
3. 服务器仅执行导入、审计、训练

本地构建离线数据包：

```bash
bash scripts/build_offline_data_bundle.sh
```

产物：
- 目录：`artifacts/offline_bundle/run_<UTC时间>/`
- 压缩包：`artifacts/offline_bundle/offline_data_bundle_<UTC时间>.tar.gz`
- 清单：`bundle_manifest.json`（含文件行数与 SHA256）

服务器导入离线包（两种方式）：

方式 A：本地直传并远程导入（`scp + ssh`）：

```bash
bash scripts/server_import_offline_bundle.sh \
  --host <server-host> \
  --port <server-port> \
  --user <server-user> \
  --remote-dir <repo-on-server> \
  --db-url "postgresql://monitor:***@localhost:5432/monitor" \
  --bundle-tar artifacts/offline_bundle/offline_data_bundle_<UTC时间>.tar.gz
```

方式 B：服务器已通过 GitHub 获取 bundle 后本机导入：

```bash
bash scripts/import_offline_data_bundle.sh \
  --bundle-dir <bundle-dir-on-server> \
  --database-url "postgresql://monitor:***@localhost:5432/monitor" \
  --primary-timeframe 5m
```

说明：
- 默认 `STRICT_ASOF_AFTER_IMPORT=0`，即导入阶段 `validate_asof_alignment` 只告警不阻断（尚未构建特征时会失败属预期）。
- 若你要把导入流程设为强门禁，可设置 `STRICT_ASOF_AFTER_IMPORT=1`。

固定顺序：
1. 历史数据完整性
2. 时间对齐
3. 数据修复
4. 特征重建
5. 训练管线
6. 模拟盘
7. 前端总控

### 4.1 Phase 1：2018-01-01 到当前完整性审计

```bash
python3 scripts/audit_full_history_completeness.py \
  --start 2018-01-01T00:00:00Z \
  --timeframe 5m \
  --symbols BTC,ETH,SOL
```

准入条件：
- `summary.history_window_complete=true`
- `summary.comment_ratio_ge_10x=true`

### 4.2 Phase 2：不达标即修复

推荐（服务器可直接执行，一键编排市场/辅助模态/社媒补采与复审计）：

```bash
bash scripts/remediate_liquid_data_gaps.sh
```

本地“全量采集 + 就绪审计 + 离线包”推荐入口（适配国内服务器离线导入）：

```bash
bash scripts/collect_required_data.sh
```

该脚本会串行执行：
- `remediate_liquid_data_gaps`（5m 主窗口）
- `1h` 市场数据补采
- 衍生品增强信号补采（长短仓比/主动买卖比/基差）
- 事件/社媒分块回填与导入
- `audit_full_history_completeness + audit_training_data_completeness(5m/1h) + audit_required_data_readiness`
- 离线 bundle 打包（可关闭）

可选环境变量（示例）：
- `START_TS=2018-01-01T00:00:00Z`
- `TIMEFRAME=5m`
- `SYMBOLS=BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK`
- `RUN_MARKET=1 RUN_AUX=1 RUN_ORDERBOOK_PROXY=1 RUN_SOCIAL=1`
- `MARKET_FALLBACK_SOURCE=none`（`1h` 场景可改 `coingecko`）
- `SOCIAL_STRICT=0`（社媒缺源时不阻断其余修复步骤）
- 国内网络推荐：`EVENT_DISABLE_GDELT=1 EVENT_DAY_STEP=30`（降低事件回填超时风险）
- 若 Reddit 限流明显：`EVENT_SOCIAL_SOURCES=reddit REDDIT_FETCH_COMMENTS=0 REDDIT_LIMIT=80`（评论比率由脚本内回填逻辑补齐）

脚本产物：
- `artifacts/data_remediation/run_<UTC时间>/audit_before.json`
- `artifacts/data_remediation/run_<UTC时间>/aux_signals.json`
- `artifacts/data_remediation/run_<UTC时间>/orderbook_proxy.json`
- `artifacts/data_remediation/run_<UTC时间>/audit_after.json`
- `artifacts/data_remediation/run_<UTC时间>/training_audit_after.json`

补充：独立执行辅助模态回填（funding + OI proxy）：

```bash
python3 scripts/ingest_binance_aux_signals.py \
  --start 2018-01-01T00:00:00Z \
  --timeframe 5m \
  --symbols BTC,ETH,SOL
```

补充：独立执行衍生品增强信号回填（2026+ regime 推荐）：

```bash
python3 scripts/ingest_binance_derivatives_signals.py \
  --start 2018-01-01T00:00:00Z \
  --period 5m \
  --symbols BTC,ETH,SOL
```

说明：
- 脚本会自动处理交易所历史窗口限制（默认约近 30 天）。
- 低频衍生品指标会自动扩展到目标 `period`（如 `5m`）以保证训练窗口覆盖。
- 当源侧缺失 `annualized_basis_rate` 时，会基于 `basis_rate` 派生补齐（用于覆盖与门禁）。

补充：独立执行订单簿代理回填（冷启动）：

```bash
python3 scripts/backfill_orderbook_from_market_bars.py \
  --start 2018-01-01T00:00:00Z \
  --timeframe 5m \
  --symbols BTC,ETH,SOL
```

兼容旧流程（按轮次审计/修复）：

```bash
python3 scripts/remediate_full_history.py \
  --start 2018-01-01T00:00:00Z \
  --timeframe 5m \
  --symbols BTC,ETH,SOL
```

修复顺序固定：
- 市场数据
- 订单簿/资金费率/链上
- 社媒帖子
- 社媒评论（目标比率 `>=10`）

### 4.3 Phase 3：as-of 对齐防泄露

```bash
python3 scripts/validate_asof_alignment.py
```

准入条件：
- `future_leakage_count=0`
- 训练快照时间与样本时间一致率 `100%`

可选：运行统一“训练可用性 + 模态覆盖 + 衍生品覆盖”审计：

```bash
python3 scripts/audit_required_data_readiness.py \
  --start 2018-01-01T00:00:00Z \
  --primary-timeframe 5m \
  --secondary-timeframe 1h \
  --symbols BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK
```

## 5. 服务器启动

### 5.1 预检

```bash
bash scripts/server_preflight.sh
```

首次在新机器部署，建议先安装运行依赖：

```bash
sudo apt-get update
sudo apt-get install -y screen postgresql redis-server
bash scripts/ensure_postgres_access.sh
```

本地无 `screen` 时可先执行轻量预检：

```bash
REQUIRE_SCREEN=0 RUN_TESTS=0 RUN_SECURITY_VALIDATION=0 bash scripts/server_preflight.sh
```

### 5.2 启动

```bash
bash scripts/server_up.sh
```

说明：
- 启动脚本使用 `screen` 拉起 `backend/collector/task_worker/model_ops/(ops_loop)`
- 脚本内会执行 Alembic 迁移
- 若系统 `python3` 依赖不全，可在 `.env` 设置 `PYTHON_BIN=/path/to/python`（默认优先使用 `.venv/bin/python`）

### 5.3 就绪检查

```bash
bash scripts/server_readiness.sh
```

### 5.4 停止

```bash
bash scripts/server_down.sh
```

### 5.5 发布/回滚预演（本地先演练）

```bash
bash scripts/rehearse_server_release_rollback.sh
```

说明：
- 默认 `DRY_RUN=1`，只打印执行序列，不真正启停服务。
- 全量预演可切 `DRY_RUN=0`；若只想演练流程不跑训练，保持 `PHASE_D_BUNDLE_DRY_RUN=1`。
- `DRY_RUN=0` 的完整演练需要系统已安装 `screen`（`server_up/server_down` 依赖）。

## 6. 特征与训练

### 6.1 特征构建

```bash
python3 scripts/build_feature_store.py --start 2018-01-01T00:00:00Z
python3 scripts/build_text_latent_embeddings.py --start 2018-01-01T00:00:00Z
python3 scripts/build_market_latent_embeddings.py --start 2018-01-01T00:00:00Z
python3 scripts/merge_feature_views.py --start 2018-01-01T00:00:00Z
```

### 6.2 训练与评估

```bash
python3 training/train_multimodal.py --start 2018-01-01T00:00:00Z
python3 training/eval_multimodal_oos.py
python3 training/register_candidate_model.py
```

Multi-horizon liquid 训练（窗口优先，生产默认不使用硬 `limit=4000`）：

```bash
python3 training/main.py \
  --run-once \
  --enable-liquid 1 \
  --liquid-train-mode production \
  --liquid-lookback-days 365 \
  --liquid-start 2025-01-01T00:00:00Z \
  --liquid-end 2026-02-20T00:00:00Z
```

可选 B 方案（多子模型 + meta 聚合）：

```bash
LIQUID_MULTI_HORIZON_TRAIN_MODE=multi_model_meta \
python3 training/main.py --run-once --enable-liquid 1 --liquid-train-mode production
```

快速迭代模式（才使用 limit/max_samples）：

```bash
python3 training/main.py \
  --run-once \
  --enable-liquid 1 \
  --liquid-train-mode fast \
  --liquid-limit 4000 \
  --liquid-max-samples 4000
```

训练收益门禁（建议每轮训练后执行）：

```bash
python3 scripts/gate_training_profitability.py \
  --track liquid \
  --lookback-hours 168 \
  --min-completed-runs 4
```

一键 multi-horizon 升级检查与产物汇总（建议每次合并前执行）：

```bash
python3 scripts/run_multi_horizon_upgrade_bundle.py --run-full-pytest
```

上线前建议严格模式（DB gates 计入 required）：

```bash
python3 scripts/run_multi_horizon_upgrade_bundle.py \
  --run-full-pytest \
  --require-db-gates \
  --strict
```

产物：
- `artifacts/upgrade/multi_horizon_upgrade_bundle_latest.json`

双卡 A100 NVLINK 训练发车（服务器）：

```bash
bash scripts/launch_dual_a100_training.sh \
  --require-a100 \
  --require-nvlink \
  --nproc 2 \
  --epochs 24
```

说明：
- 该脚本会检查 GPU 数量、型号和 NVLINK 状态（可通过 `--allow-non-a100/--allow-no-nvlink` 放宽）。
- 内部调用 `scripts/train_gpu_stage2.py --compute-tier a100x2 --nproc-per-node 2`。

多模态消融评估（去文本 / 去宏观 / 事件窗口）：

```bash
python3 training/eval_multimodal_oos.py \
  --ablations full,no_text,no_macro,event_window \
  --event-strength-threshold 0.05 \
  --out artifacts/models/multimodal_eval.json
```

Phase-D 一键执行（推荐服务器直接跑）：

```bash
bash scripts/run_phase_d_multimodal_bundle.sh
```

说明：
- 脚本会按顺序执行 `train_multimodal -> backbone_experiments -> eval_multimodal_oos -> register_candidate_model`。
- 所有参数都可通过 `.env` 配置（见 `.env.example` 的 `Phase-D multimodal experiment bundle` 与 `Candidate registration gate` 区段）。
- 若设置 `CANDIDATE_ENFORCE_GATES=1`，注册门禁失败会返回非零退出码，方便 CI/调度器直接阻断。

无泄露验证可接入多模态候选门禁（可选强制）：

```bash
python3 scripts/validate_no_leakage.py \
  --track liquid \
  --lookback-days 180 \
  --require-candidate-gate-passed 1 \
  --candidate-max-age-hours 72 \
  --require-multimodal-gate-passed 1 \
  --multimodal-gate-max-age-hours 24
```

说明：
- 候选门禁读取 `artifacts/models/candidate_registry.jsonl` 最新记录。
- 调度器门禁读取 `artifacts/ops/multimodal_gate_state.json`。
- 两项都可在 `.env` 通过 `VALIDATE_*` / `MULTIMODAL_GATE_*` 开关配置。

多骨干统一对照（iTransformer / PatchTST / Ridge，同一 WF/PKF 协议）：

```bash
python3 training/backbone_experiments.py \
  --start 2018-01-01T00:00:00Z \
  --backbones ridge,itransformer,patchtst,tft \
  --out artifacts/experiments/backbone_suite_latest.json
```

说明：
- 本地若缺少 `torch`，`itransformer/patchtst` 会在报告中标记 `blocked: torch_missing`。
- 服务器定时训练可通过 `TRAIN_ENABLE_BACKBONE_EXPERIMENTS=1` 开启该实验套件（默认关闭，不影响生产训练流程）。

## 7. 模拟盘与人工实盘控制

### 7.1 持续模拟盘守护

```bash
python3 monitoring/paper_trading_daemon.py --loop --interval-sec 60 \
  --execution-events-file artifacts/paper/paper_execution_events.jsonl
```

连续运维循环（含首训后再开模拟盘 + 收益门禁失败自动微调）：

```bash
python3 scripts/continuous_ops_loop.py --loop
```

说明：
- 默认 `OPS_PAPER_AFTER_FIRST_TRAIN=1`：首次训练成功前跳过模拟盘下单。
- 默认 `OPS_PROFIT_GATE_CMD` 会执行 `gate_training_profitability.py`。
- 默认 `OPS_RUN_FINETUNE_ON_GATE_FAIL=1`：收益门禁失败触发 `OPS_FINETUNE_CMD` 微调流程。

### 7.2 监控接口

- `GET /api/v2/monitor/history-completeness`
- `GET /api/v2/monitor/alignment`
- `GET /api/v2/monitor/social-throughput`
- `GET /api/v2/monitor/model-status`
- `GET /api/v2/monitor/multimodal-health`
- `GET /api/v2/monitor/paper-performance`
- `GET /api/v2/monitor/horizon-performance`
- `GET /api/v2/monitor/prediction-drift`
- `GET /api/v2/monitor/confidence-calibration`
- `GET /api/v2/monitor/paper-pnl-buckets`

### 7.3 Multi-horizon 预测与信号

`/api/v2/predict/liquid` 已支持返回多 horizon 输出（`1h/4h/1d/7d`）：
- `expected_return`（map）
- `signal_confidence`（map）
- `vol_forecast`（map）
- `score_horizons` / `edge_horizons` / `action_horizons`
- 兼容字段：`expected_return_legacy`、`signal_confidence_legacy`、`vol_forecast_legacy`

`/api/v2/signals/generate`（liquid）已升级为：
- `edge_h = expected_return_h - cost_h`
- `score_h = edge_h / max(vol_h, eps)`
- 阈值可按 horizon 配置：`SIGNAL_SCORE_ENTRY_BY_HORIZON`、`SIGNAL_CONFIDENCE_MIN_BY_HORIZON`

### 7.4 控制接口

- `POST /api/v2/control/live/enable`
- `POST /api/v2/control/live/disable`

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v2/control/live/enable \
  -H 'Content-Type: application/json' \
  -d '{"operator":"admin","reason":"manual_enable_live","paper_enabled":false}'

curl -X POST http://127.0.0.1:8000/api/v2/control/live/disable \
  -H 'Content-Type: application/json' \
  -d '{"operator":"admin","reason":"manual_disable_live","paper_enabled":true}'
```

模型 registry（symbol+horizon）人工切换与回滚：

```bash
curl http://127.0.0.1:8000/api/v2/models/liquid/registry

curl -X POST http://127.0.0.1:8000/api/v2/models/liquid/registry/candidate \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC","horizon":"1h","model_name":"liquid_baseline_btc","model_version":"v2.1-candidate","operator":"ops"}'

curl -X POST http://127.0.0.1:8000/api/v2/models/liquid/registry/promote-candidate \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC","horizon":"1h","operator":"ops"}'

curl -X POST http://127.0.0.1:8000/api/v2/models/liquid/registry/activate \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC","horizon":"1h","model_name":"liquid_baseline_btc","model_version":"v2.0","operator":"ops"}'

curl -X POST http://127.0.0.1:8000/api/v2/models/liquid/registry/rollback \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC","horizon":"1h","operator":"ops"}'

python3 scripts/rollback_liquid_model.py --symbol BTC --horizon 1h --operator ops
```

## 8. 前端发布

### 8.1 构建

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

### 8.2 运行方式

可选：
- 直接由静态服务器托管 `frontend/dist`
- 或由 Nginx 托管并反向代理 `/api` 到 `127.0.0.1:8000`

## 9. 升级流程（建议）

```bash
git pull
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
bash scripts/bootstrap_env.sh
bash scripts/server_preflight.sh
bash scripts/server_down.sh
bash scripts/server_up.sh
bash scripts/server_readiness.sh
```

若门禁失败：
- 先修复数据完整性/对齐问题
- 禁止直接进入训练和实盘

## 10. 回滚流程（建议）

```bash
bash scripts/server_down.sh
git checkout <last-stable-tag-or-commit>
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
bash scripts/server_up.sh
bash scripts/server_readiness.sh
```

## 11. 常见问题

`[FAIL] missing command: screen`
- 安装：`sudo apt install -y screen`

预检通过但启动后 API 不可用
- 查看日志：`/tmp/backend_screen.log`、`/tmp/task_worker_screen.log`

完整性审计不达标
- 先执行 `scripts/remediate_full_history.py`
- 修复后重跑 `scripts/audit_full_history_completeness.py`

实盘误开关担忧
- 系统默认人工控制
- 仅 `POST /api/v2/control/live/enable` 会开启实盘
