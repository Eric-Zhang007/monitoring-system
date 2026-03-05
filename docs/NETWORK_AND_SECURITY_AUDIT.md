# Network / Proxy / Security Audit (Strict-Only)

## Scope
- 目标：在不破坏 strict-only 主链路前提下，补齐网络可达性、代理支持、安全审计、UTC 漂移门禁、风控邮件可视化。
- 约束：不引入 withdraw/transfer；live 启动 fail-fast；关键动作可追审计。

## Audit Matrix

| Capability | Status | Evidence (file:line) | Call Chain | Risk | Patch (minimal diff) |
|---|---|---|---|---|---|
| Proxy profile 管理（global/account/process） | DONE | `scripts/init_db.sql:736`, `scripts/init_db.sql:753`, `backend/v2_repository.py:3428`, `backend/v2_repository.py:3518`, `backend/v2_router.py:4559`, `backend/v2_router.py:4601` | API `/api/v2/ops/proxy_profiles|proxy_bindings` -> `V2Repository.upsert_proxy_profile/upsert_proxy_binding` -> `proxy_profiles/proxy_profile_bindings` | 无法按账号/进程分流网络，live 可达性不可控 | 新增 DB 表+CRUD+绑定解析，保持原交易链路不改 |
| Connectivity probe（REST+WS） | DONE | `backend/connectivity_service.py:35`, `backend/connectivity_service.py:69`, `backend/connectivity_service.py:91`, `backend/v2_router.py:4521`, `backend/v2_router.py:4531`, `backend/v2_repository.py:3364` | `/api/v2/ops/connectivity/probe` -> `ConnectivityService.probe` -> `repo.save_venue_connectivity_status` | 无法区分网络问题 vs 策略问题；live 盲启动 | 新增探测服务、状态落库和查询 API |
| live 启动前连通性 hard gate | DONE | `backend/v2_router.py:4384`, `backend/v2_router.py:4411` | `/api/v2/ops/process/start`(LIVE_TRADER) -> connectivity probe -> fail-fast | 网络不可达仍下单，触发连续拒单/429 | 启动前强校验 `rest_ok && ws_ok`，失败 503 |
| UTC/时钟漂移检测 + live gate | DONE | `backend/clock_drift_service.py:24`, `backend/clock_drift_service.py:42`, `backend/v2_router.py:4415`, `backend/v2_repository.py:3717`, `scripts/init_db.sql:789` | `/api/v2/ops/process/start`(LIVE_TRADER) -> `ClockDriftService.probe` -> red 则 fail-fast | 时钟偏差导致签名/订单时序异常 | 新增 drift 探测与落库；red 阻断 live |
| RBAC 最小权限（viewer/operator/admin） | DONE | `backend/rbac.py:9`, `backend/rbac.py:29`, `backend/v2_router.py:4331`, `backend/v2_router.py:4563`, `backend/v2_router.py:4760` | 关键 API -> `require_role()` -> 403/放行 | 弱权限可改密钥/风控配置 | 在敏感路由加 role 门禁；默认可通过 `RBAC_ENFORCE` 控制 |
| 审计日志（敏感操作） | DONE | `scripts/init_db.sql:766`, `backend/v2_repository.py:3616`, `backend/v2_router.py:299` | 路由动作 -> `_audit_action` -> `repo.save_audit_log` -> `audit_logs` | 事后不可追责/不可复盘 | 统一审计写入 + 掩码 |
| Secrets 加密存储与不回显 | DONE | `scripts/init_db.sql:779`, `backend/secrets_manager.py:13`, `backend/v2_router.py:4677`, `backend/v2_router.py:4756`, `backend/v2_router.py:4549` | `/live/accounts` & `/ops/secrets` -> encrypted at rest; list API strips secret fields | 明文泄漏导致实盘账户风险 | 使用 Fernet 加密；列表接口不返回密文字段 |
| 风控邮件推送与投递日志 | DONE | `backend/mailer.py:21`, `backend/mailer.py:35`, `backend/mailer.py:86`, `backend/v2_router.py:292`, `backend/v2_router.py:4671`, `scripts/init_db.sql:715` | 风险事件/命令 -> `_notify_risk_mail` -> `send_risk_email` -> `mail_delivery_logs` | 关键风险无人知晓 | 邮件成功/失败均落库并可查询 |
| 严格命令语法（风控命令） | DONE | `backend/risk_command_center.py:13`, `backend/risk_command_center.py:44`, `backend/risk_command_center.py:112`, `backend/v2_router.py:4698` | `/api/v2/risk/commands/execute` -> `parse_risk_command` -> `RiskCommandExecutor.execute` -> logs + optional mail | 命令语义不明确导致误操作 | 白名单语法 + parse/execute 双回执 |
| 前端可视化（网络/代理/风控邮件） | DONE | `frontend/src/pages/LiveMonitorPage.tsx:56`, `frontend/src/pages/LiveMonitorPage.tsx:66`, `frontend/src/pages/ConfigCenterPage.tsx:46`, `frontend/src/pages/ConfigCenterPage.tsx:220`, `frontend/src/pages/RiskCenterPage.tsx:85` | UI -> 对应 API 拉取/操作 -> 状态面板刷新 | 运维无可视化，排障慢 | 在既有页面补状态卡、探测、邮件日志面板 |
| 禁止 withdraw/transfer 路径 | DONE | `backend/v2_router.py:4752`, `frontend/src/pages/LiveMonitorPage.tsx:56`, `rg -n "withdraw|transfer"` 结果 | 代码层未实现提币/划转 API | 资金安全边界被破坏 | 明确声明无该能力，且无相关路由实现 |

## Notes on Existing vs New
- 关键执行主链（strict 工件校验、对账/kill switch、序列输入）未重写，仅在控制平面增加前置门禁与可观测性。
- 连通性/时钟/代理增强均通过新增服务和路由接入，原 `/api/v2/predict/*` 与执行核心路径未破坏。

## Tests Added / Verified
- `tests/test_connectivity_probe_mocked.py`
- `tests/test_connectivity_probe_with_proxy_profile.py`
- `tests/test_live_start_blocked_when_unreachable.py`
- `tests/test_clock_drift_blocks_live.py`
- `tests/test_proxy_profile_binding.py`
- `tests/test_rbac_permissions.py`
- `tests/test_secrets_not_returned.py`
- `tests/test_email_risk_notification.py`
- `tests/test_init_db_has_control_plane_tables.py`

本地回归结果：`17 passed`（网络/代理/安全/控制平面相关测试集合）。

## Residual Risks (明确保留)
1. WS 探测在未安装 `websockets` 依赖时使用 TCP 连接探测（`backend/connectivity_service.py:69`），可用性高但不做完整 WS handshake。生产建议安装 `websockets` 以获得完整握手探测。
2. 运行中进程的代理切换目前通过“新连接/重启进程”生效；`process` 级绑定建议 `requires_restart=true`，前端已提供重启入口。

## Why This Should Not Break Strict Logic
- 所有新增 gate 仅作用于 live 启动前检查，失败直接 `HTTP 503`，不存在 silent fallback。
- 密钥与审计为附加控制面，不修改模型训练/推理与订单执行核心算法。
- 新表均为附加表，不改现有主表字段，不影响旧查询语义。
