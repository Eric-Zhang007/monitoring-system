# CODEX Strict-Only Production Audit Report

## Scope & Method
- Scope: strict-only production readiness audit for model/execution/data链路，按 A1-A7 逐项核对，并只做最小差异修复。
- Method: `rg`/逐文件审阅 + 关键单测执行（不做重构）。

## Data-Side 2018+ Completeness Check
- 尝试直接连本地 PostgreSQL 检查关键表最早时间（`market_bars/events/feature_snapshots/feature_matrix_main/funding_rates/orderbook_l2/text_embeddings`）。
- 结果：当前环境无法完成真实性核验（`DATABASE_URL=postgresql://monitor@localhost:5432/monitor` 返回 `fe_sendauth: no password supplied`），因此此项在本地仅能给出“检查方法已落地、数据真实性待有权限环境执行”。

---

## Audit Matrix (A1-A7)

### A1) Artifact “clean clone 可训练/可推理”
- Status: **DONE (fixed in this PR)**
- Evidence:
  - `.gitignore:89`-`.gitignore:91` 忽略 `artifacts/**`（仅放行 `.gitkeep`/`README.md`）。
  - 训练入口改为稳定包导入：`training/train_liquid.py:17` -> `from mlops_artifacts.pack import pack_model_artifact`。
  - 服务仓储改为稳定包导入：`backend/v2_repository.py:17` -> `from mlops_artifacts.validate import validate_manifest_dir`。
  - 新增稳定包：`mlops_artifacts/pack.py:18`、`mlops_artifacts/validate.py:20`。
  - 调用链：
    - `training/train_liquid.py:374` `main()` -> `pack_model_artifact()`（`training/train_liquid.py:506`）
    - `backend/v2_repository.py` 模型注册/读取校验 -> `validate_manifest_dir()`（如 `backend/v2_repository.py:952`）
- Risk:
  - 若继续从 `artifacts/` 导入 Python 代码，clean clone 会直接 `ImportError`，训练/服务不可启动。
- Patch Plan:
  - 保留 `artifacts/` 只存运行产物；导入逻辑迁移到 `mlops_artifacts/`，并更新所有引用。
  - 新增 clean-clone 导入测试 `tests/test_clean_clone_imports.py:9`。

### A2) 内部状态闭环（fills -> stats -> account_state -> risk/sizer/style）
- Status: **DONE**
- Evidence:
  - `AccountStateAggregator` 读取执行统计并落库：
    - 读取 `recent_execution_stats`: `backend/account_state/aggregator.py:84`-`backend/account_state/aggregator.py:93`
    - 写 `balances_state`/`positions_state`/`account_state_snapshots`: `backend/account_state/aggregator.py:115`-`backend/account_state/aggregator.py:149`
  - `RiskManager` 使用 execution_stats：
    - `reject_rate_5m` 硬阈值: `backend/risk_manager.py:60`
    - `slippage_bps_p90` 软惩罚: `backend/risk_manager.py:72`-`backend/risk_manager.py:76`
  - `ExecutionStyleSelector` 使用 exec_stats + 风格惩罚：
    - `slippage_high` 与 `exec_style_bias`：`backend/execution_style_selector.py:66`-`backend/execution_style_selector.py:79`
  - 调用链（入口到策略动作）：
    - `/signals/generate` -> `_get_account_state_for_signal()`：`backend/v2_router.py:2556`、`backend/v2_router.py:2593`
    - `RiskManager.evaluate(...)`：`backend/v2_router.py:2595`
    - `ExecutionStyleSelector.select_style(...)`：`backend/v2_router.py:2615`
  - 覆盖测试：
    - `tests/test_signal_pipeline_uses_account_state.py:67`、`tests/test_signal_pipeline_uses_account_state.py:83`
    - `tests/test_risk_manager_soft_penalties.py:15`
    - `tests/test_execution_style_selector.py:8`
- Risk:
  - 若链路断开，风控与执行风格不会感知真实成交质量，易产生过激下单与成本劣化。
- Patch Plan:
  - 已有实现与测试覆盖充分，本 PR 不重写，仅保留并验证。

### A3) Reconciliation + kill switch
- Status: **DONE**
- Evidence:
  - `run_once()` 包含关键动作：
    - `poll_order`：`monitoring/reconciliation_daemon.py:95`
    - `fetch_fills`：`monitoring/reconciliation_daemon.py:103`
    - `update_parent_from_fills`：`monitoring/reconciliation_daemon.py:105`
    - 持仓漂移检测：`monitoring/reconciliation_daemon.py:142`-`monitoring/reconciliation_daemon.py:147`
    - 写 `risk_events`：`monitoring/reconciliation_daemon.py:149`
    - 触发 kill switch：`monitoring/reconciliation_daemon.py:159`、`monitoring/reconciliation_daemon.py:190`
  - 覆盖测试：
    - `tests/test_reconciliation_triggers_red.py:74`
    - `tests/test_recon_failure_triggers_kill_switch.py:66`
- Risk:
  - 若缺失对账闭环，实盘仓位漂移无法及时止损，风控审计链路断裂。
- Patch Plan:
  - 不重写主逻辑，保留现状并验证测试通过。

### A4) 执行端幂等 + 限频/backoff
- Status: **PARTIAL (补强测试，未做架构重写)**
- Evidence:
  - 重试/backoff 逻辑：
    - Coinbase `_request`：`backend/execution_engine.py:452`-`backend/execution_engine.py:466`
    - Bitget `_request`：`backend/execution_engine.py:1028`-`backend/execution_engine.py:1040`
  - client_order_id 幂等缓存：
    - Coinbase：`backend/execution_engine.py:544`-`backend/execution_engine.py:547`、`backend/execution_engine.py:580`
    - Bitget：`backend/execution_engine.py:1126`-`backend/execution_engine.py:1129`、`backend/execution_engine.py:1205`
  - 新增强验收测试：
    - `tests/test_live_adapter_idempotency.py:18`（429 重试）
    - `tests/test_live_adapter_idempotency.py:44`（5xx 重试）
    - `tests/test_live_adapter_idempotency.py:69`（Coinbase 幂等）
    - `tests/test_live_adapter_idempotency.py:102`（Bitget 幂等）
- Risk:
  - 当前幂等缓存是进程内内存语义，进程重启后幂等信息不可回放，极端情况下存在重复提交风险。
- Patch Plan:
  - 先以最小 diff 补“强验收测试”锁定当前语义；跨进程幂等（持久化）留作后续扩展，不在本轮做重构。

### A5) Universe 管理（训练/推理一致 + 可回放）
- Status: **PARTIAL (本 PR 完成最小落地)**
- Evidence:
  - DB 表已补齐：
    - `scripts/init_db.sql:438` 创建 `asset_universe_snapshots`
    - 索引 `scripts/init_db.sql:448`
  - 仓储方法：
    - `upsert_asset_universe_snapshot(...)`：`backend/v2_repository.py:746`
    - `resolve_asset_universe_asof(...)`：`backend/v2_repository.py:791`
  - 训练接入与记录：
    - 参数：`training/train_liquid.py:76`-`training/train_liquid.py:77`
    - 解析 universe：`training/train_liquid.py:148`
    - 写 snapshot：`training/train_liquid.py:173`
    - 报告写入 universe 字段：`training/train_liquid.py:190`、`training/train_liquid.py:496`
  - backtest 路径也支持 asof resolve：`backend/v2_router.py:2123`-`backend/v2_router.py:2130`
  - 新增测试：
    - `tests/test_init_db_has_universe_table.py:6`
    - `tests/test_train_report_records_universe.py:9`
- Risk:
  - 若 universe 不可追溯，训练/回放一致性下降，模型评估与实盘解释不可对齐。
- Patch Plan:
  - 本轮已完成“最小可用”落地（表 + 训练记录 + 测试）。
  - 推理/decision_trace 中 universe 元信息统一落库属于下一步增强，不在本轮做破坏性改动。

### A6) 动态成本口径一致（含 funding）
- Status: **PARTIAL**
- Evidence:
  - 统一成本核心函数：`cost/cost_profile.py:74` `compute_cost_bps(...)`
  - Label 写入成本：`training/labels/liquid_labels.py:8`、`training/labels/liquid_labels.py:40`
  - 数据集将 `cost_bps` 进入训练样本：`training/datasets/liquid_sequence_dataset.py:50`、`training/datasets/liquid_sequence_dataset.py:143`
  - 动作层成本映射：`backend/v2_router.py:1868`-`backend/v2_router.py:1908`
  - funding_rates 表存在：`scripts/init_db.sql:414`
  - 但 `compute_cost_bps` 当前未显式纳入 funding 成本项（无 funding 参数）。
- Risk:
  - 对 perp 交易场景，忽略 funding 可能导致收益预估偏乐观，影响策略效益。
- Patch Plan:
  - 保持默认行为不变（最小破坏），本轮先在审计中标注为可选增强项；如启用再以 env 开关接入。

### A7) label 与执行延迟对齐（lag）
- Status: **MISSING (optional)**
- Evidence:
  - `compute_label_targets(...)` 直接使用 `p1 = prices[index + step]`：`training/labels/liquid_labels.py:35`
  - 未见 lag 开关或 fill-lag 对齐逻辑。
- Risk:
  - 回测标签可能略乐观（未反映真实成交延迟），收益估计偏差风险上升。
- Patch Plan:
  - 维持默认不变，建议后续用显式开关实现（`ENABLE_LABEL_FILL_LAG`），避免影响现有基线。

---

## Implemented Fixes In This PR

### Hard Tasks
- B1 (P0): artifacts 导入修复
  - 新包：`mlops_artifacts/__init__.py`、`mlops_artifacts/pack.py`、`mlops_artifacts/validate.py`
  - 全量替换 `artifacts.pack/validate` 引用为 `mlops_artifacts.*`
  - 新增测试：`tests/test_clean_clone_imports.py`

- B2: universe snapshot 落地
  - 新增表：`scripts/init_db.sql` 中 `asset_universe_snapshots`
  - 新增仓储 upsert：`backend/v2_repository.py:746`
  - 训练接入：`training/train_liquid.py` 增加 `--universe-track` 与 `--use-universe-snapshot`
  - 报告追踪：`training_report.universe`
  - 新增测试：`tests/test_init_db_has_universe_table.py`、`tests/test_train_report_records_universe.py`

### Optional Task Triggered by Audit Gap
- C3: 执行端幂等/backoff 强验收
  - 新增 `tests/test_live_adapter_idempotency.py`
  - 并加入 `scripts/run_strict_e2e_acceptance.sh`

---

## Why These Fixes Won’t Break Existing Logic
- strict-only 默认行为保持不变：
  - 未引入 v1/v2/v3 分支、未引入静默回退。
  - 训练/推理产物路径仍是 `artifacts/models/...`，只迁移了“可导入代码位置”。
- 变更策略是“门禁 + 兼容入口”：
  - universe snapshot 通过参数控制，默认开启但保留显式关闭（用于单测/离线场景）。
  - 现有模型结构、训练损失、推理输出协议未被重写。
- 测试补强覆盖关键新增点：
  - clean clone 导入
  - DB schema 包含 universe 表
  - training_report 回放信息
  - live adapter retry/idempotency

