# Frontend Acceptance Checklist

## 本地启动
```bash
cd frontend
npm install
npm run dev
```
默认连接 `VITE_API_URL`（未配置时 `http://localhost:8000`）。

## 页面清单
- `/dashboard` 总览
- `/offline-training` 离线训练监管
- `/paper` 模拟盘监管
- `/live` 实盘监管（多账号）
- `/config` 配置中心
- `/process` 进程控制台
- `/risk` 风控中心 + 命令

## 手动验收
1. 总览页显示运行进程、风险事件数量、固定成本 `1.88 CNY / hour`。
2. 离线训练页点击“运行离线审计”，可看到 task 创建与状态变化。
3. 离线训练页点击“启动训练”，进程控制台能看到 `TRAIN_LIQUID` 任务。
4. 模拟盘页可修改 `paper.initial_usdt` 并成功调用 `/api/v2/config`（无需重启）。
5. 实盘页可新增 Bitget 账号（前端不回显 secret），并能执行连通性探测。
6. 配置中心保存 `requires_restart=true` 配置后，能通过“一键重启”触发 `/api/v2/ops/process/restart`。
7. 进程控制台可 Start/Stop/Restart，日志与 metrics 持续刷新。
8. 风控中心执行命令：
   - `CMD KILL_SWITCH ON track=liquid strategy=global`
   - `CMD KILL_SWITCH OFF track=liquid strategy=global`
   - `CMD RESTART PROCESS id=<pid>`
   可看到 parse/execute 回执与审计日志。
9. 风控中心展示 `risk_events` 与 `reconciliation_logs`。
10. UI 全链路中不存在 withdraw/transfer 按钮或接口入口。

## 失败判据
- 任一核心页面白屏或 API 500 未显式报错。
- 命令执行返回 parse 成功但无审计记录。
- `requires_restart=true` 配置无法引导到进程重启。
- 实盘页面出现提币/划转相关能力入口。
