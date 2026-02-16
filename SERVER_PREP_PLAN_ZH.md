# 上服务器前准备与部署计划（AutoDL/国内机房）

## 1. 目标
- 在不依赖远端公网拉取镜像的前提下，完成本地打包、上传、服务器启动、数据库恢复与验收。
- 保证部署链路可重复执行，且每一步都有明确脚本入口。
- 目标服务器默认按“无 Docker（screen + python）”流程执行；Docker 链路仅作为可选备用。

## 2. 前置条件
- 本地（笔记本/WSL）已安装并可用：`python3`、`git`、`ssh`、`scp`（Docker 仅在备用链路需要）。
- 服务器已安装并可用：`python3`、`git`、`screen`、`postgresql`、`redis-server`。
- 仓库代码与本地验证通过。

## 3. 脚本清单
- `scripts/server_preflight.sh`
- `scripts/server_preflight_nodocker.sh`
- `scripts/server_readiness_nodocker.sh`
- `scripts/server_package_images.sh`
- `scripts/server_upload_bundle.sh`
- `scripts/server_seed_db.sh`
- `scripts/server_bootstrap.sh`
- `scripts/server_verify_runtime.sh`
- `scripts/train_gpu_stage2.py`
- `scripts/ingest_bitget_market_bars.py`
- `scripts/import_market_bars_csv.py`
- `scripts/build_multisource_events_2025.py`
- `scripts/import_events_jsonl.py`
- `scripts/prepare_training_data_2025_now.sh`
- `scripts/server_import_2025_data.sh`
- `scripts/seed_liquid_universe_snapshot.py`
- `scripts/audit_training_data_completeness.py`
- `scripts/server_nodocker_up.sh`
- `scripts/server_nodocker_down.sh`

## 4. 标准执行流程

### Step A: 无 Docker 预检（主流程）
```bash
bash scripts/server_preflight_nodocker.sh
```

### Step A2: 严格预检（要求在线 API）
```bash
REQUIRE_LIVE_API=1 REQUIRE_DB=1 bash scripts/server_preflight_nodocker.sh
```
输出中重点检查：
- `disk_gb` / `mem_gb` 达标
- 目标测试通过（task queue/health/task worker）
- `torch_probe={"skipped":"gpu_not_required"}`（no-GPU 目标）

### Step B: 启动无 Docker 运行栈
```bash
bash scripts/server_nodocker_up.sh
```

### Step C: 无 Docker 运行时验收（完整非 GPU pipeline）
```bash
bash scripts/server_readiness_nodocker.sh
```

### Step D: （可选备用）本地打包镜像与配置（Docker 方案）
```bash
bash scripts/server_package_images.sh --bundle-dir artifacts/server_bundle/latest
```
输出：
- `artifacts/server_bundle/latest/`
- `artifacts/server_bundle/latest.tar.gz`

### Step E: （可选）导出数据库快照
```bash
bash scripts/server_seed_db.sh export --out artifacts/server_bundle/monitor.fc
```

### Step F: 上传到服务器
```bash
bash scripts/server_upload_bundle.sh \
  --bundle artifacts/server_bundle/latest.tar.gz \
  --db-dump artifacts/server_bundle/monitor.fc \
  --host <HOST> --port <PORT> --user root \
  --remote-dir /opt/monitoring-system/bundles
```

### Step G: 服务器侧启动（Docker 备用链路，建议 screen）
```bash
screen -S deploy-monitor
bash /opt/monitoring-system/current/scripts/server_bootstrap.sh \
  --bundle /opt/monitoring-system/bundles/latest.tar.gz \
  --deploy-dir /opt/monitoring-system/current \
  --db-dump /opt/monitoring-system/bundles/monitor.fc
```

### Step H: 服务器侧验收（Docker 备用链路）
```bash
bash /opt/monitoring-system/current/scripts/server_verify_runtime.sh \
  --deploy-dir /opt/monitoring-system/current
```

## 5. 无 Docker 运行流程（非 GPU 目标）

### Step N1: 代码与依赖
```bash
git clone https://github.com/Eric-Zhang007/monitoring-system.git
cd monitoring-system
python3 -m pip install -U pip
python3 -m pip install -r backend/requirements.txt
grep -vE '^(torch|torchvision)==?' inference/requirements.txt > /tmp/infer_req_no_torch.txt
grep -vE '^(torch|torchvision)==?' training/requirements.txt > /tmp/train_req_no_torch.txt
python3 -m pip install -r /tmp/infer_req_no_torch.txt
python3 -m pip install -r /tmp/train_req_no_torch.txt
```

### Step N2: 启动核心服务（backend + collector + task_worker + model_ops）
```bash
# 推荐先显式固定运行时语义（无 Docker 默认）
export MODEL_DIR=/path/to/monitoring-system/backend/models
export FEATURE_VERSION=feature-store-v2.1
export FEATURE_PAYLOAD_SCHEMA_VERSION=v2.2
export DATA_VERSION=v1
export COST_FEE_BPS=5.0
export COST_SLIPPAGE_BPS=3.0
export COST_IMPACT_COEFF=120.0
# 生产 run_source=prod + strict_asof 时，将禁止 legacy fallback 并 fail-fast
bash scripts/server_nodocker_up.sh
```

### Step N2.2: 运行时 readiness 验收
```bash
bash scripts/server_readiness_nodocker.sh
```

### Step N2.3: 固定 as-of 资产池快照（防前视偏差）
```bash
python3 scripts/seed_liquid_universe_snapshot.py \
  --track liquid \
  --symbols BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK \
  --as-of 2025-01-01T00:00:00Z \
  --version top10_static_v1 \
  --source pretrain_seed
```

### Step N2.4: 数据完整性审计
```bash
python3 scripts/audit_training_data_completeness.py \
  --symbols BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK \
  --lookback-days 420 \
  --timeframe 5m
```

### Step N2.5: 服务器无法直连交易所时（本地拉数据再导入）
本地一键生成 2025 至今训练数据包（Top10 5m + 1h + 多信源事件 + 社媒样本）：
```bash
bash scripts/prepare_training_data_2025_now.sh
```

开机后一键上传并导入服务器（支持 `sshpass`）：
```bash
SSHPASS='<SERVER_PASSWORD>' \
HOST=connect.bjb1.seetacloud.com PORT=40111 USER_NAME=root \
bash scripts/server_import_2025_data.sh
```

该脚本会执行：
- 上传 `market_bars_top10_5m_2025_now.csv`
- （可选）上传 `market_bars_top10_1h_2025_now.csv`
- 上传 `events_multisource_2025_now.jsonl`
- （可选）上传并导入 `social_history_2025_now.jsonl`
- 依次执行 `import_market_bars_csv.py`、`import_events_jsonl.py`、`import_social_events_jsonl.py`
- 运行 `audit_training_data_completeness.py --lookback-days 420 --timeframe 5m`
- 输出 events 时间覆盖范围 + source mix 快照

### Step N3: 后台状态查看
```bash
screen -ls
tail -f /tmp/backend_screen.log
tail -f /tmp/task_worker_screen.log
```

## 6. 运行建议
- 长任务（训练/回测/部署）统一放在 `screen` 会话中执行，避免 SSH 断开导致任务中断。
- 每次部署后保留以下证据：
  - `manifest.json`
  - `docker compose ps`
  - 验收脚本输出
- 若仅更新后端逻辑而不变更镜像基座，可使用 `--skip-build` 进行快速打包验证。

## 7. 回滚策略
- 保留上一个 bundle（镜像 tar + compose + 配置）目录，不覆盖。
- 回滚时重新执行 `server_bootstrap.sh` 指向上一个 bundle。
- DB 回滚使用上一版 `pg_dump` 快照，通过 `server_seed_db.sh import` 恢复。
