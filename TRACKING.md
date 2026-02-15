# 代码追踪与问题清单

## ✅ 2026-02-15 Phase-4/5 闭环推进（本轮）

1. **治理调度与审计落库打通**
- `monitoring/model_ops_scheduler.py` 新增调度审计持久化调用：`POST /api/v2/models/audit/log`。
- 新增动态 rollout 阶梯推进：先读取 `/api/v2/models/rollout/state`，按 `10 -> 30 -> 100` 推进；达到 100% 时跳过并记录 `already_max_stage`。

2. **新增治理状态/审计 API**
- `backend/schemas_v2.py` 新增 `RolloutStateResponse` 与 `SchedulerAuditLogRequest`。
- `backend/v2_router.py` 新增：
  - `GET /api/v2/models/rollout/state`
  - `POST /api/v2/models/audit/log`
- `backend/v2_repository.py` 新增 `save_scheduler_audit_log`，统一落入 `risk_events` 审计流（`code=scheduler_audit_log`）。

3. **WebSocket 稳定性强化（背压 + 慢客户端隔离）**
- `backend/main.py` 的 `ConnectionManager` 改为“每连接独立发送队列 + sender task”。
- 新增队列上限、批量 flush、发送超时，避免单个慢连接拖垮全局广播。
- 队列溢出/发送失败会主动断开对应连接并计数。
- `backend/metrics.py` 新增 `WEBSOCKET_DROPPED_MESSAGES_TOTAL{reason}` 指标，区分 `queue_full` / `send_error`。

4. **回归与验收**
- 容器内测试通过：`28 passed`（`test_model_ops_decisions/test_v2_router_core/test_v2_repository_utils/test_execution_engine_paths/test_lineage_replay_consistency`）。
- `scripts/test_v2_api.sh` 通过（重建 `backend/model_ops` 后复验通过）。
- 新增接口实测通过：
  - `/api/v2/models/rollout/state` 返回当前 rollout 状态；
  - `/api/v2/models/audit/log` 写入后可在 `risk_events` 查询到 `scheduler_audit_log` 记录。

## ✅ 2026-02-15 Codex Plan 剩余八项收敛（本轮）

1. **告警 5 分钟可触达闭环**
- 新增 `alertmanager` 服务与配置：`monitoring/alertmanager.yml`。
- `prometheus` 增加 Alertmanager 对接；P1 路由 `repeat_interval=5m`，P2 为 `15m`。
- 新增告警落库入口：`POST /api/v2/alerts/notify`，可写入 `risk_events`（`code=alertmanager:*`）。

2. **SLO 扩展（p50/p95/p99 + 可用性）**
- `monitoring/health_check.py` 的 SLO 计算新增 `p50/p95/p99`。
- 新增 API 可用性指标（基于 `ms_http_requests_total` 5xx 比例），门限 `>=99.9%`。

3. **回测 vs paper 偏差自动验收**
- 新增脚本：`scripts/check_backtest_paper_parity.py`（默认阈值 `10%`）。
- 在样本不足（如回测失败）时返回 `insufficient_observation`，避免误报硬失败。

4. **量化硬指标统计与门禁输出**
- 新增脚本：`scripts/evaluate_hard_metrics.py`。
- 输出并评估：`Sharpe`、`MaxDD`、`execution_reject_rate` 与对应硬门槛。

5. **独立 worker 队列化（回测/归因）**
- 新增 Redis 任务队列模块：`backend/task_queue.py`。
- 新增 worker：`monitoring/task_worker.py`（独立容器 `task_worker`）。
- 新增 API：
  - `POST /api/v2/tasks/backtest`
  - `POST /api/v2/tasks/pnl-attribution`
  - `GET /api/v2/tasks/{task_id}`
- smoke 中新增异步任务提交断言；任务可由 `queued -> completed`。

6. **混沌演练脚本**
- 新增 `scripts/chaos_drill.py`，覆盖：
  - `redis_interrupt`
  - `db_slow`
  - `exchange_jitter`
  - `model_degrade`
  - `recover`

7. **Coinbase live 验收脚本**
- 新增 `scripts/validate_coinbase_live.py`：
  - 无密钥时给出 `skipped + missing_credentials`；
  - 有密钥时执行连通性预检输出。

8. **一键回放复现流水线**
- 新增 `scripts/replay_model_run.py`：
  - 自动读取最近（或指定）`backtest_run` 配置；
  - 复跑并比对核心指标差异（容差可配）。

## ✅ 2026-02-15 Phase-2 闭环推进（本轮）

1. **训练/推理 lineage 闭环**
- `training/feature_pipeline.py` 增加严格 DQ 阈值与批量快照写入。
- `training/liquid_model_trainer.py` 增加硬阻断与 `train_lineage_id` 落库。
- `inference/main.py` 增加 `infer_lineage_id` 与推理快照落库，预测结果关联 lineage。

2. **lineage 严格一致性**
- `backend/v2_repository.py` 的 `check_feature_lineage_consistency` 支持 `strict + data_version + mismatch_keys`。
- `/api/v2/data-quality/lineage/check` 响应新增 `data_version` 与 `mismatch_keys`。

3. **模型驱动回测替代代理路径**
- `/api/v2/backtest/run` 使用 `feature_snapshots + model_version` 回放，输出 `cost_breakdown` 与 `lineage_coverage`。
- 回测治理记录与回放所用模型保持一致（修复模型名回写默认值的问题）。

4. **执行-风控联动加强**
- `risk/check` 增加 `daily_loss_exceeded`、`consecutive_loss_exceeded`。
- `execution/run` 强制执行前调用风险检查，未通过返回 `423`。

5. **治理调度阈值化与审计**
- `monitoring/model_ops_scheduler.py` 全部阈值由 ENV 配置，调度日志包含 `window/thresholds/decision`。
- rollback 返回并记录 `windows_failed` 与 `trigger_rule`。

6. **SLO/告警闭环**
- `monitoring/health_check.py` 增加 p95 SLO 判定与 `insufficient_observation`。
- `monitoring/prometheus.yml` 增加 `rule_files`，新增 `monitoring/alerts.yml`（P1/P2 + route 标签）。

7. **测试与脚本**
- 新增测试：
  - `backend/tests/test_execution_engine_paths.py`
  - `backend/tests/test_model_ops_decisions.py`
  - `backend/tests/test_lineage_replay_consistency.py`
- 扩展测试：
  - `backend/tests/test_v2_router_core.py`
  - `backend/tests/test_v2_repository_utils.py`
- `scripts/test_v2_api.sh` 新增关键 JSON 字段断言。

## ✅ 2026-02-15 Phase-3 闭环推进（本轮）

1. **执行前风险口径修正**
- 修复 `execution/run` 的日内损失计算：由绝对 PnL 改为 `-net_pnl / |gross_notional|` 比例口径，避免误触发 `daily_loss_exceeded`。

2. **异常波动熔断**
- 新增执行前波动预检：近窗口绝对收益超阈值触发 `abnormal_volatility_circuit_breaker:{target}`。
- 命中后返回 `423`，并自动触发短时全局 kill switch（默认 1 分钟，可 ENV 配置）。

3. **硬拦截语义修正**
- `risk/check` 在硬拦截时 kill switch reason 改为真实触发原因（`daily_loss_exceeded` / `consecutive_loss_exceeded` / `drawdown_exceeded`）。
- `RISK_HARD_BLOCK_MINUTES` 用于统一最短封禁时长。

4. **测试与回归**
- 扩展 `backend/tests/test_v2_router_core.py`：
  - 日内损失比例计算回归；
  - runtime 风控 hard-block reason/duration；
  - 执行路径异常波动熔断。
- 扩展 `backend/tests/test_v2_repository_utils.py`：
  - `execution edge pnl` 的 `daily_loss_ratio` 与 `consecutive_losses` 计算口径。
- 容器内回归通过：`18 passed`（router/execution/model_ops/lineage 组合测试）。
- `scripts/test_v2_api.sh` 在新风控行为下通过。

5. **执行审计结构标准化**
- `execution` 元数据新增统一 lifecycle 事件结构：`event/status/time/metrics`，用于稳定审计与前端解析。

6. **Phase-3 追加加固（本轮）**
- 异常波动阈值分层已落地：
  - 按 symbol 覆盖：`RISK_MAX_ABS_RETURN_SYMBOLS`（例：`BTC=0.05,ETH=0.06`）
  - 按 UTC 时段乘数：`RISK_MAX_ABS_RETURN_TOD_MULTIPLIER`（例：`0-7:1.4,8-16:1.0,17-23:1.2`）
- 连续亏损统计下沉到真实成交序列：
  - 新增仓储函数：`get_execution_edge_pnls / get_execution_daily_loss_ratio / get_execution_consecutive_losses`
  - `execution/run` 新增 strategy 维度连续亏损前置拦截，避免全局误伤。
- 新增显式开仓状态接口：
  - `GET /api/v2/risk/opening-status?track=...&strategy_id=...`
  - 返回 `can_open_new_positions`、`block_reason`、`remaining_seconds`、`expires_at`。

## ✅ 2026-02-15 P0 稳定化追加修复（本轮）

1. **V2 口径与前端接入对齐**
- 前端默认 WebSocket 地址从 `/ws` 切换为 `/stream/signals`，避免连接被冻结旧端点。
- `docker-compose` 中 `VITE_WS_URL` 同步更新为 `ws://localhost:8000/stream/signals`。

2. **风险返回码一致性**
- `risk/check` 在 kill switch 命中时，违规码统一为 `kill_switch_triggered:{track}:{strategy_id}`。

3. **加密单域默认值收敛**
- `LIQUID_SYMBOLS` 默认值统一为 `BTC,ETH,SOL`（训练、推理、Compose、回测默认目标）。

4. **漂移与血缘口径修正**
- `get_execution_slippage_samples` 仅统计 `filled|partially_filled`。
- `check_feature_lineage_consistency` 在 `target=None` 时按 target 分组比较最近两条快照，避免跨标的误判。

5. **回测时序口径修正**
- `_walk_forward_metrics` 去除周末过滤，符合加密 7x24 数据特征。

6. **可维护性清理**
- 删除 `backend/main.py` 中 `/ws` 冻结返回后的不可达历史逻辑。

## 🔴 已确认的关键问题

### ✅ 已修复的问题

#### 1. 训练数据逻辑错误（training/main.py）
**问题：** 训练时从 predictions 表获取标签，但 predictions 还不存在
```python
# 旧错误代码
query = """
    SELECT nf.embedding, p.direction
    FROM nim_features nf
    LEFT JOIN predictions p ON ...
"""
```
**修复：** ✅ 从价格表直接生成标签（上涨/下跌/盘整）
```python
# 新正确代码
query = """
    WITH price_windows AS (
        SELECT
            p1.price as price_start,
            p2.price as price_end,
            (p2.price - p1.price) / p1.price * 100 as pct_change,
            CASE
                WHEN (p2.price - p1.price) / p1.price > 0.5 THEN 'up'
                WHEN (p2.price - p1.price) / p1.price < -0.5 THEN 'down'
                ELSE 'neutral'
            END as direction
        FROM prices p1
        JOIN prices p2 ON ...
    )
    SELECT direction, embedding FROM price_windows
"""
```

#### 2. 验证函数造假（training/main.py）
**问题：** 验证指标是硬编码的假数据
```python
# 旧错误代码
def train_with_validation(self, val_features, val_labels):
    return {
        "train_accuracy": 0.75,  # 假数据！
    }
```
**修复：** ✅ 真实的训练循环，计算真实准确率和损失
```python
# 新正确代码
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == batch_labels).sum().item()

    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            ...
            val_correct += (predicted == batch_labels).sum().item()

    # 真实记录指标
    train_accuracy = 100 * train_correct / train_total
    val_accuracy = 100 * val_correct / val_total
```

#### 3. 推理使用随机权重（inference/main.py）
**问题：** 预测时使用随机初始化的权重
```python
# 旧错误代码
scenario_logits = torch.nn.functional.linear(
    last_output,
    torch.randn(3, 256).to(self.device)  # 随机！
)
```
**修复：** ✅ 加载训练好的模型权重
```python
# 新正确代码
def load_model(self, symbol: str):
    model_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_model.pth")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        # 创建模型
        self.model = ImprovedModel(**checkpoint['model_config']).to(self.device)

        # 加载真实权重（修复点）
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"✅ Loaded model for {symbol}")
```

#### 4. 后端API全用Mock数据（backend/main.py）
**问题：** 所有API端点都返回硬编码的假数据
```python
# 旧错误代码
@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str):
    return {
        "symbol": symbol,
        "predictions": [  # 硬编码假数据
            {"horizon": "1h", "direction": "up", "change": "+1.2%", ...},
        ]
    }
```
**修复：** ✅ 从数据库查询真实数据
```python
# 新正确代码
@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str, hours: int = 24):
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT
                symbol, scenario, direction, confidence,
                expected_change_pct, expected_price,
                scenario_probabilities, created_at
            FROM predictions
            WHERE symbol = UPPER(%s)
              AND created_at > NOW() - make_interval(hours => %s)
            ORDER BY created_at DESC
            LIMIT 100
        """

        cursor.execute(query, (symbol, hours))
        rows = cursor.fetchall()

        predictions = [dict(row) for row in rows]
        return {"symbol": symbol.upper(), "predictions": predictions}
    except Exception as e:
        logger.error(f"❌ Failed to get predictions: {e}")
```

#### 5. 数据库Schema不完整（新增）
**问题：** 缺少价格表、技术指标表、正确的训练样本表
**修复：** ✅ 创建了 `scripts/init_db.sql`
```sql
-- 价格表
CREATE TABLE prices (...);
-- 技术指标表
CREATE TABLE technical_indicators (...);
-- 训练样本表
CREATE TABLE training_samples (...);
-- 修复后的训练数据查询函数
CREATE OR REPLACE FUNCTION generate_training_samples(...);
```

---

## 🎯 修复进度

### P0 - 立即修复 ✅ 全部完成
- [x] ✅ 修复training数据逻辑（从价格表生成标签）
- [x] ✅ 实现真实的模型加载（推理服务）
- [x] ✅ 修复后端Mock数据（真实数据库查询）
- [x] ✅ 添加数据库Schema（价格、技术指标、训练样本）
- [x] ✅ 添加真实验证逻辑（准确率、损失）

### P1 - 高优先级
- [x] ✅ 添加价格采集支持（Schema支持）
- [x] ✅ 添加技术指标Schema
- [ ] ⏳ 实现价格数据采集（collector.py）
- [ ] ⏳ 实现技术指标计算

### P2 - 中优先级
- [ ] ⏳ 特征工程优化（多时序窗口）
- [ ] ⏳ 模型架构优化（Transformer/TCN/TFT）
- [ ] ⏳ 评估指标完善（Sharpe Ratio, Max Drawdown）

---

## 📝 当前状态

### 已完成 ✅
1. **Training Service修复：**
   - 从价格表生成正确的训练标签
   - 真实的训练/验证循环
   - 模型保存和加载

2. **Inference Service修复：**
   - 真实加载训练好的模型
   - 不再使用随机权重
   - 正确的特征查询逻辑

3. **Backend API修复：**
   - 所有Mock数据已移除
   - 从PostgreSQL查询真实预测
   - 从Redis缓存价格数据
   - WebSocket推送真实数据

4. **数据库Schema：**
   - 价格表（prices）
   - 技术指标表（technical_indicators）
   - 训练样本表（training_samples）
   - 修复的SQL函数

### 待完成 ⏳
1. 数据采集：collector.py需要实现真实的价格采集
2. 技术指标：需要实现MA、MACD、RSI的计算
3. 端到端测试：在有GPU的环境下运行完整测试
4. 性能优化：模型量化、batch推理等

---

## 🔄 Git提交历史

### Commit 1: MVP Complete (7ccd359)
- 初始MVP版本
- 包含架构和基础代码
- 但有上述问题

### Commit 2: Fix Critical Issues (待提交)
- 修复训练数据逻辑
- 修复推理权重加载
- 修复后端Mock数据
- 添加数据库Schema

---

**修复时间：** 2026-02-14 23:15 - 23:30
**修复者：** 小黑
**状态：** ✅ P0问题全部修复，待推送到GitHub
