# CODEX Implementation Notes

## Scope
- Strict-only production readiness uplift for:
  - offline training data audit
  - runtime dynamic config
  - process manager control plane
  - risk command parsing/execution + email notification hooks
  - frontend no-code control console

## Progress
- [x] Baseline repository scan and capability inventory
- [x] DB schema extensions
- [x] Offline data audit script + API + persistence
- [x] Runtime config service + hot reload API
- [x] Process manager + lifecycle APIs + logs/metrics
- [x] Email risk command parser/executor + audit
- [x] Frontend console routes/pages
- [x] Tests and strict acceptance script updates
- [x] Documentation updates

## Hot Reload vs Restart
- 热更新（无需重启）：
  - `runtime_config` 普通项（`requires_restart=false`）
  - `paper.initial_usdt`、风险阈值类、样本/策略模板类参数
  - 命令 `CMD SET CONFIG ...` 默认热生效
- 需要重启（前端会标注并提供一键重启）：
  - `requires_restart=true` 的 runtime 配置项
  - 典型包括：端口、数据库连接、权重路径、进程启动命令参数
  - 前端经 `/api/v2/ops/process/restart` 触发对应进程重启

## Known Risks
- Existing backend has both legacy monitor APIs and v2 APIs; new control plane endpoints must avoid path collision.
- Offline audit must remain lightweight for synchronous trigger fallback, while still allowing async task execution.
- 邮件发送依赖 SMTP 环境变量；若未配置会记录失败日志并返回明确错误原因。
- 全量 `pytest -q` 在当前仓库存在历史测试导入冲突（`training/metrics` 与顶层 `metrics` 命名冲突、旧测试依赖缺失导出）；本轮新增与 strict 验收集测试均已通过。
