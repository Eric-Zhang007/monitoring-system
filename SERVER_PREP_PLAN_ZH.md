# 上服务器前准备与部署计划（AutoDL/国内机房）

## 1. 目标
- 在不依赖远端公网拉取镜像的前提下，完成本地打包、上传、服务器启动、数据库恢复与验收。
- 保证部署链路可重复执行，且每一步都有明确脚本入口。

## 2. 前置条件
- 本地（笔记本/WSL）已安装并可用：`docker`、`docker compose`、`ssh`、`scp`。
- 服务器已安装并可用：`docker`、`docker compose`、`screen`。
- 仓库代码与本地验证通过。

## 3. 脚本清单
- `scripts/server_preflight.sh`
- `scripts/server_package_images.sh`
- `scripts/server_upload_bundle.sh`
- `scripts/server_seed_db.sh`
- `scripts/server_bootstrap.sh`
- `scripts/server_verify_runtime.sh`

## 4. 标准执行流程

### Step A: 本地预检
```bash
bash scripts/server_preflight.sh
```

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

## 5. 运行建议
- 长任务（训练/回测/部署）统一放在 `screen` 会话中执行，避免 SSH 断开导致任务中断。
- 每次部署后保留以下证据：
  - `manifest.json`
  - `docker compose ps`
  - 验收脚本输出
- 若仅更新后端逻辑而不变更镜像基座，可使用 `--skip-build` 进行快速打包验证。

## 6. 回滚策略
- 保留上一个 bundle（镜像 tar + compose + 配置）目录，不覆盖。
- 回滚时重新执行 `server_bootstrap.sh` 指向上一个 bundle。
- DB 回滚使用上一版 `pg_dump` 快照，通过 `server_seed_db.sh import` 恢复。
