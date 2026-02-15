# 上服务器前准备与部署计划（AutoDL/国内机房）

## 1. 目标
- 在不依赖远端公网拉取镜像的前提下，完成本地打包、上传、服务器启动、数据库恢复与验收。
- 保证部署链路可重复执行，且每一步都有明确脚本入口。
- 若服务器内核受限导致 `dockerd` 不可用，提供可直接切换的“无 Docker（conda + screen + torchrun）”流程。

## 2. 前置条件
- 本地（笔记本/WSL）已安装并可用：`docker`、`docker compose`、`ssh`、`scp`。
- 服务器已安装并可用：`docker`、`docker compose`、`screen`。
- 仓库代码与本地验证通过。

## 3. 脚本清单
- `scripts/server_preflight.sh`
- `scripts/server_preflight_nodocker.sh`
- `scripts/server_package_images.sh`
- `scripts/server_upload_bundle.sh`
- `scripts/server_seed_db.sh`
- `scripts/server_bootstrap.sh`
- `scripts/server_verify_runtime.sh`
- `scripts/train_gpu_stage2.py`

## 4. 标准执行流程

### Step A: 本地预检
```bash
bash scripts/server_preflight.sh
```

### Step A2: 服务器受限时的无 Docker 预检
```bash
MIN_GPU_COUNT=2 REQUIRE_DB=0 bash scripts/server_preflight_nodocker.sh
```
输出中重点检查：
- `torch_probe.cuda_available=true`
- `torch_probe.cuda_device_count>=2`
- `gpu_count>=2`

### Step B: 本地打包镜像与配置
```bash
bash scripts/server_package_images.sh --bundle-dir artifacts/server_bundle/latest
```
输出：
- `artifacts/server_bundle/latest/`
- `artifacts/server_bundle/latest.tar.gz`

### Step C: （可选）导出数据库快照
```bash
bash scripts/server_seed_db.sh export --out artifacts/server_bundle/monitor.fc
```

### Step D: 上传到服务器
```bash
bash scripts/server_upload_bundle.sh \
  --bundle artifacts/server_bundle/latest.tar.gz \
  --db-dump artifacts/server_bundle/monitor.fc \
  --host <HOST> --port <PORT> --user root \
  --remote-dir /opt/monitoring-system/bundles
```

### Step E: 服务器侧启动（建议 screen）
```bash
screen -S deploy-monitor
bash /opt/monitoring-system/current/scripts/server_bootstrap.sh \
  --bundle /opt/monitoring-system/bundles/latest.tar.gz \
  --deploy-dir /opt/monitoring-system/current \
  --db-dump /opt/monitoring-system/bundles/monitor.fc
```

### Step F: 服务器侧验收
```bash
bash /opt/monitoring-system/current/scripts/server_verify_runtime.sh \
  --deploy-dir /opt/monitoring-system/current
```

## 5. 无 Docker 训练/推理流程（受限服务器）

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

### Step N2: 双卡训练（screen + torchrun）
```bash
screen -S train-stage2
cd /path/to/monitoring-system
export TRAIN_RUN_ONCE=1
export TRAIN_ENABLE_VC=1
export TRAIN_ENABLE_LIQUID=1
export LIQUID_SYMBOLS=BTC,ETH,SOL
python3 scripts/train_gpu_stage2.py --compute-tier a100x2 --nproc-per-node 2 --enable-vc --enable-liquid
```

### Step N3: 后台状态查看
```bash
screen -ls
screen -r train-stage2
tail -f artifacts/gpu_stage2/train_gpu_stage2_latest.json
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
