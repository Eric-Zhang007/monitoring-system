# 量化系统实施规范（真实数据优先，包含模型升级路径）

## 1. 范围与目标

本文定义了解决当前系统问题所需的精确工程改动：

1. 门控（gating）中 `prod` 与 `maintenance` 数据污染
2. Sharpe 指标失真及统计可比性差
3. 短窗口调参导致的过拟合风险
4. 数据源广度不足与来源可靠性问题
5. 策略路径规则过重，模型驱动 alpha 不足
6. 超参数搜索效率低（串行网格）
7. 信息质量与模型决策价值缺乏硬性关联

目标状态：
- 门控决策仅基于真实数据（`run_source=prod`, `score_source=model`）
- 离线/在线双轨清晰且可审计
- 模型路径可执行：线性/树基线 -> 神经网络 -> 集成 -> 分阶段发布
- HPO 默认采用并行贝叶斯优化（Optuna），而非串行暴力网格

---

## 2. 不可协商原则（不得更改）

1. 真实数据门控原则
- 发布门控只能使用 `prod + model` 运行。
- `maintenance/smoke/async_test` 仅用于监控。

2. 可复现原则
- 同数据 + 同配置 + 同代码 => 同结果（在数值容差内）。
- 核心回测逻辑中不得存在仅由隐藏环境变量触发的行为。

3. 反过拟合原则
- 禁止单窗口“优化后直接宣称有效”。
- 必须包含 Search/Validation/Forward-OOS 分段。

4. 指标透明原则
- 必须披露 Sharpe 计算方法（`daily_agg` vs `step_raw`）。
- 禁止只报告有利的指标口径。

5. 数据血缘原则
- 每条训练与推理样本都必须可追溯到来源、时间戳、映射方法和质量分。

6. 风险优先原则
- 收益提升不能破坏硬性风控约束（最大回撤、拒单率、执行约束）。

7. 生产安全原则
- 新模型/策略必须具备影子发布（shadow）/金丝雀发布（canary）和回滚路径。

8. 合规原则
- 仅使用公开信息，不得使用内幕数据。

---

## 3. 当前系统诊断（仓库已验证）

## 3.1 真实数据与维护数据混用
观察：
- `backtest_runs` 在近期窗口中包含大量 `maintenance` 样本。
- 门控脚本历史上默认使用 `prod,maintenance` 混合过滤。

影响：
- 门控结果无法纯粹代表真实市场状态。

## 3.2 Sharpe 失真
观察：
- 回测引擎按 step 级别计算 Sharpe，并使用高频年化（`sqrt(24*365)`）。

影响：
- 数值量级不稳定且容易被夸大。

## 3.3 规则主导的策略路径
观察：
- `backend/v2_router.py` 中信号 -> 仓位 -> 换手/成本的核心执行仍以规则为中心。
- 当前 liquid 产物路径常落到轻量 tabular/json 模型。

影响：
- 模型贡献无法与启发式策略行为清晰隔离。

## 3.4 数据源薄弱
观察：
- GDELT 连接器经常因 429 失败。
- Macro FRED 数据源常为空。
- 数据来源集中于有限通道。

影响：
- 特征多样性与新鲜度鲁棒性不足，难以稳定产生 alpha。

## 3.5 调参效率低
观察：
- 默认调参脚本主要为串行网格循环。

影响：
- 探索效率差，耗时高，更容易拟合特定市场阶段。

---

## 4. 架构目标：离线/在线双轨

## 4.1 离线轨（研究与训练）
职责：
- 数据整理与质量过滤
- 特征生成与 schema 强约束
- 模型训练与 OOS 验证
- 超参数优化

产出：
- 版本化产物（`model`, `normalization`, `manifest`）
- 具有固定窗口与血缘 ID 的验证报告

## 4.2 在线轨（推理与执行）
职责：
- 严格 schema 的实时特征计算
- 具备回退记账的模型推理
- 风险约束下的下单生成
- 监控与漂移检测

产出：
- 预测、决策、执行记录、质量指标

## 4.3 离线/在线对齐契约
必须共享字段：
- `feature_payload_schema_version`
- `feature_version`
- `data_version`
- `model_name`, `model_version`
- `lineage_id`

若存在 schema/版本不匹配，禁止晋升发布。

---

## 5. 具体实施计划

## Track A：真实数据门控强制化

### A1. 默认过滤器（已改，保持锁定）
文件：
- `backend/v2_router.py`
- `scripts/evaluate_hard_metrics.py`
- `scripts/check_backtest_paper_parity.py`
- `docker-compose.yml`
- `scripts/daily_phase63_maintenance.sh`

行为：
- 默认包含来源：`prod`
- 默认排除来源：`smoke,async_test,maintenance`

### A2. 在运行配置/指标中新增 `data_regime`
文件：
- `backend/schemas_v2.py`
- `backend/v2_router.py`

改动：
- 扩展回测配置，加入 `data_regime` 枚举：
  - `prod_live`
  - `maintenance_replay`
  - `mixed`
- 将该字段写入 `backtest_runs.config`，并在门控脚本中反映。

验收：
- 门控命令输出包含 `data_regime`，且默认拒绝非 `prod_live`。

---

## Track B：指标方法加固（Sharpe 与观测充分性）

### B1. 引入双 Sharpe 输出
文件：
- `backend/v2_router.py`
- `scripts/evaluate_hard_metrics.py`

改动：
- 在回测指标输出中新增：
  - `sharpe_step_raw`
  - `sharpe_daily`
  - `sharpe_method="daily_agg_v1"`
  - `observation_days`
  - `vol_floor_applied`
- 门控脚本必须只评估 `sharpe_daily`。

### B2. 日度聚合实现
实现规则：
- 将 step 级 pnl 按 UTC 日桶聚合。
- 用日收益计算 Sharpe，年化因子为 `sqrt(365)`。
- `step_raw` 仅用于诊断。

### B3. 观测充分性检查
门控条件变为：
- `completed_runs >= min_completed_runs`
- `observation_days >= min_observation_days`
- `data_regime == prod_live`

验收：
- 天数/样本不足时必须返回 `status=insufficient_observation`。

---

## Track C：数据源扩展与可靠性

### C1. 连接器可靠性框架
文件：
- `collector/connectors/gdelt.py`
- `collector/connectors/macro_fred.py`
- `collector/collector.py`
- `backend/metrics.py`

改动：
- 增加连接器级重试（指数退避 + 抖动）。
- 增加来源健康计数：
  - success/failure
  - empty_result_count
  - rate_limit_count
  - fetch_latency
- 对重复失败来源增加 cooldown/circuit breaker。

### C2. 来源分层与质量门控
文件：
- `collector/collector.py`
- `training/feature_pipeline.py`
- `inference/main.py`

改动：
- 在事件载荷中附加来源质量分层与置信度。
- 特征流水线按层级加权聚合事件。
- 可通过配置排除低质量事件参与训练。

### C3. 延迟优先的采集目标
SLO 目标：
- 主要加密数据源 `P95(source_publish_to_ingest) < 120s`
- 主要连接器成功率 > 95%

验收：
- 仪表盘必须展示每个来源的新鲜度与错误画像。

---

## Track D：从策略到模型的升级路径（可执行、分阶段）

## D0. 基线冻结（当前状态）
基线模型类别：
- 表格模型线性/LightGBM 产物（`liquid_*_lgbm_baseline_v2.json`）
- 可选 TSMixer checkpoint（`liquid_*_tsmixer_v2.pt`）

要求：
- 将基线冻结为后续所有比较的对照组。

## D1. 强化表格模型阶段（线性/树做规范）
文件：
- `training/liquid_model_trainer.py`
- `training/validation.py`
- `inference/model_router.py`

实现：
- 用显式模型注册表替换临时回退加权。
- 使用 purged K-fold + walk-forward 报告训练 LightGBM。
- 持久化完整训练元数据：
  - 数据区间
  - 特征 schema
  - 各 fold 指标
  - seed

晋升规则：
- 树模型必须在验证集与前瞻 OOS 上均优于冻结基线。

## D2. 神经网络阶段（TSMixer）+ 师生蒸馏
文件：
- `training/liquid_model_trainer.py`
- `inference/model_router.py`
- `backend/models/*` manifest 格式

实现：
- 教师：已验证的表格模型预测。
- 学生：基于序列特征的 TSMixer。
- 蒸馏目标：`MSE(student, true)` + `lambda * MSE(student, teacher)`。
- 保存 checkpoint 时必须包含字段：
  - `type=tsmixer_liquid`
  - `normalization`
  - `n_tokens`, `n_channels`
  - `ensemble_alpha`
  - `train_report_hash`

晋升规则：
- 同时评估 NN-only 与 Ensemble。
- 仅当集成模型提升扣费后净收益且不恶化最大回撤约束时才可接受。

## D3. 集成阶段与发布
文件：
- `inference/model_router.py`
- `backend/v2_router.py`
- `backend/v2_repository.py`

实现：
- 显式集成策略版本：
  - `tabular_only`
  - `nn_only`
  - `blend_alpha_x`
- 将所选策略写入回测配置与预测解释。
- 发布策略：
  - shadow -> 10% canary -> 分阶段提高流量

回滚触发器：
- 连续门控失败
- 漂移阈值超限
- 拒单/滑点异常

---

## Track E：超参数优化升级（并行 + Optuna）

### E1. 新 HPO 脚本
新增文件：
- `scripts/optuna_liquid_hpo.py`

关键设计：
- 采样器：`TPESampler`
- 剪枝器：`MedianPruner`（或等效 ASHA）
- 存储：SQLite（`artifacts/hpo/optuna_liquid.db`）
- 并行 worker 默认：`max(1, cpu_count()-2)`

目标函数（多项标量化）：
- 最大化 `pnl_after_cost`
- 惩罚 `max_drawdown`
- 惩罚 `reject_rate`
- 加入 `turnover` 惩罚

### E2. 三阶段优化协议
1. 阶段 1：粗搜（单交易对、短窗口）
2. 阶段 2：候选精修（多交易对、中窗口）
3. 阶段 3：前瞻 OOS 确认（长窗口、仅 prod）

仅阶段 3 胜出者可成为门控候选。

### E3. 增量持久化
- 每个 trial 立即写入一行 JSONL。
- 长任务中断后必须可恢复。

验收：
- HPO 可占满 `n-2` 核。
- 到达 top-k 的时间显著优于串行网格基线。

---

## Track F：信息处理流水线（不只是采集）

### F1. 事件处理层
1. Ingest（原始采集）
2. Normalize（类型/时间/来源规范化）
3. Entity-link（交易标的/公司/宏观索引实体链接）
4. Dedup cluster（去重聚类）
5. Quality scoring（质量评分）
6. Feature projection（特征投影）
7. Decision attribution（决策归因）

### F2. 质量到决策的问责闭环
每次预测必须输出：
- 主要贡献事件
- 来源分层
- 特征贡献摘要
- 缺失标记

### F3. LLM 集成边界
允许：
- 抽取、分类、摘要、实体归一化
禁止：
- 直接用于下单决策

LLM 输出必须先转为确定性的结构化特征，再供模型使用。

---

## 6. 宏观/微观错误清单与修复

## 宏观错误
1. 混合状态门控 -> 通过仅 prod 默认与 `data_regime` 修复
2. 指标方法不一致 -> 通过双 Sharpe + 日度门控修复
3. 搜索/验证泄漏 -> 通过严格三阶段窗口修复
4. 来源集中风险 -> 通过连接器可靠性 + 来源分层修复

## 微观错误
1. 仓位覆盖不一致
- 问题：`_score_to_size()` 直接读取环境变量；可能忽略请求级覆盖配置。
- 修复：将生效后的仓位配置对象贯穿整个回测调用栈。

2. 合成事件链接污染
- 问题：过宽的 symbol 回退可能以弱语义抬高覆盖率。
- 修复：标记合成链接，并默认从训练中排除。

3. 产物有效性标准过弱
- 问题：文件可解析 != 科学上有效。
- 修复：强制产物 manifest 字段 + 签名评估元数据。

4. 调参脚本鲁棒性
- 问题：串行循环、输出延迟、中断损失。
- 修复：增量持久化 + 可恢复 + Optuna 并行化。

---

## 7. 交付清单（实施项）

1. 代码改动
- [ ] 增加 `data_regime` 字段与过滤
- [ ] 增加 `sharpe_daily` 流水线并切换门控
- [ ] 增加连接器健康指标与熔断器
- [ ] 增加严格产物 manifest 校验
- [ ] 增加 Optuna HPO 脚本与运行文档
- [ ] 重构仓位配置传递，移除隐藏环境变量依赖

2. 数据/运维改动
- [ ] 仅 prod 门控看板
- [ ] 数据源健康看板
- [ ] 仅 OOS 晋升报告

3. 测试
- [ ] Sharpe 日度聚合单元测试
- [ ] 来源过滤严格性（仅 `prod`）单元测试
- [ ] 仓位覆盖确定性单元测试
- [ ] HPO 恢复与并行 worker 集成测试
- [ ] 产物校验规则回归测试

---

## 8. 验收标准

仅当以下条件全部满足时，升级才算通过：

1. 门控纯度
- 门控统计仅包含 `run_source=prod` 和 `score_source=model`。

2. 指标可靠性
- `sharpe_daily` 取代 step 原始 Sharpe 作为门控输入。
- 观测不足返回显式状态。

3. 模型升级路径可执行
- 完成 Tabular 基线 -> NN -> 集成，并有可追踪 OOS 对比。
- 仅在可量化净改进下晋升集成模型。

4. 搜索效率
- HPO 并行运行时长显著优于串行网格。
- 搜索流水线可恢复且抗中断。

5. 数据价值证明
- 引入新增信息的特征变体，相对基线在 OOS 上体现正向贡献。

---

## 9. 下一轮冲刺（执行顺序）

Sprint 1（高优先级）：
1. 实施 Sharpe 日度门控方法 + 观测充分性检查
2. 以确定性方式重构仓位配置传递
3. 增加严格产物 manifest 校验

Sprint 2：
1. 连接器可靠性/熔断器 + 数据源健康指标
2. 实施 Optuna 并行 HPO
3. 增加阶段化（Search/Validation/OOS）晋升逻辑

Sprint 3：
1. 加固基于蒸馏的 TSMixer 训练路径
2. 通过 shadow/canary 与自动回滚完成集成策略发布
3. 在决策流水线中接入信息质量贡献报告

---

## 10. 可执行任务清单（按 Sprint 拆解）

说明：本节将蓝图转换为可直接执行的工程任务。每个任务都包含目标、改动文件、实施动作、验收命令、完成定义（DoD）。

### Sprint 1（高优先级，先修可信度与可复现性）

#### S1-T1：切换为日度 Sharpe 门控 + 观测充分性
目标：门控不再使用 step Sharpe，改用 `sharpe_daily`，并对观测不足返回明确状态。

改动文件：
- `backend/v2_router.py`
- `scripts/evaluate_hard_metrics.py`
- `backend/schemas_v2.py`
- `backend/tests/test_hard_metrics_gate.py`

实施动作：
1. 在回测结果中新增字段：`sharpe_step_raw`、`sharpe_daily`、`sharpe_method`、`observation_days`、`vol_floor_applied`。
2. `scripts/evaluate_hard_metrics.py` 仅使用 `sharpe_daily` 进行阈值判断。
3. 增加 `--min-observation-days` 参数，并将不足样本统一标记为 `status=insufficient_observation`。
4. 保留 `step_raw` 仅用于诊断输出，不参与通过/失败判定。

验收命令：
- `pytest backend/tests/test_hard_metrics_gate.py -q`
- `python3 scripts/evaluate_hard_metrics.py --track liquid --lookback-days 180 --score-source model --include-sources prod --exclude-sources smoke,async_test,maintenance`

DoD：
- 输出 JSON 中存在 `sharpe_daily` 与 `observation_days`。
- 门控结果在观测不足时返回 `insufficient_observation`。
- 所有判定逻辑不再读取 `sharpe`（step 口径）。

#### S1-T2：引入 `data_regime` 并默认拒绝非 `prod_live`
目标：把“真实数据门控”从约定变成强约束。

改动文件：
- `backend/schemas_v2.py`
- `backend/v2_router.py`
- `scripts/evaluate_hard_metrics.py`
- `scripts/check_backtest_paper_parity.py`
- `backend/tests/test_metrics_gate_score_source_filter.py`

实施动作：
1. 在 `BacktestRunRequest` 增加 `data_regime` 字段（`prod_live|maintenance_replay|mixed`）。
2. 回测写入 `backtest_runs.config.data_regime`。
3. 门控脚本默认只接受 `data_regime=prod_live`（除非显式放宽）。
4. 报表输出中显示 `data_regime` 分布与有效样本数。

验收命令：
- `pytest backend/tests/test_metrics_gate_score_source_filter.py -q`
- `python3 scripts/check_backtest_paper_parity.py --track liquid --score-source model --include-sources prod --exclude-sources smoke,async_test,maintenance`

DoD：
- 默认门控结果中不包含 `maintenance_replay/mixed`。
- 人为注入非 `prod_live` 回测后，门控应拒绝并给出原因。

#### S1-T3：仓位参数传递去隐式环境依赖
目标：回测与在线推理共用同一套“显式参数”，避免 `_score_to_size()` 隐式读取 env 造成不一致。

改动文件：
- `backend/v2_router.py`
- `backend/tests/test_strategy_position_sizing.py`

实施动作：
1. 让 `_score_to_size()` 接收 `sizing_cfg`（必填）而不是内部读取环境变量。
2. 回测入口、信号生成、执行路径统一构建 `effective_sizing_cfg` 并向下传递。
3. 保留环境变量作为“默认配置加载层”，但不可在核心函数中直接读取。

验收命令：
- `pytest backend/tests/test_strategy_position_sizing.py backend/tests/test_v2_router_core.py -q`

DoD：
- 同请求参数在不同环境变量下结果一致。
- 所有 sizing 回归测试通过。

#### S1-T4：强化 artifact manifest 校验
目标：从“文件存在”升级到“结构完整且可追溯”。

改动文件：
- `backend/v2_repository.py`
- `backend/models/*.json`
- `backend/tests/test_model_ops_decisions.py`

实施动作：
1. 对 JSON artifact 增加强校验必填项：`model_name`、`model_version`、`track`、`type`、`created_at`、`feature_version`、`data_version`。
2. 对 PT artifact 增加 checkpoint 必填项检查：`type`、`normalization`、`train_report_hash`（或等效字段）。
3. 缺字段时返回 `model_artifact_invalid`，并写入原因。

验收命令：
- `pytest backend/tests/test_model_ops_decisions.py -q`

DoD：
- 缺字段 artifact 无法通过 `model_artifact_exists()`。
- 合规 artifact 可通过且不影响现有发布流程。

### Sprint 2（中优先级，修可靠性与搜索效率）

#### S2-T1：连接器可靠性与数据源健康指标
目标：实现来源级重试、熔断、健康统计，降低 429/空数据对训练与推理的冲击。

改动文件：
- `collector/connectors/gdelt.py`
- `collector/connectors/macro_fred.py`
- `collector/collector.py`
- `backend/metrics.py`
- `backend/tests/test_health_slo.py`

实施动作：
1. 为每个 connector 增加统一重试策略（指数退避 + 抖动 + 上限）。
2. 增加来源级 cooldown/circuit breaker（连续失败触发，冷却后恢复）。
3. 上报指标：`success/failure/empty_result/rate_limit/fetch_latency`。
4. 在事件 payload 增加抓取质量字段（如 `source_fetch_status`、`source_confidence`）。

验收命令：
- `pytest backend/tests/test_health_slo.py -q`
- `python3 scripts/chaos_drill.py`

DoD：
- 单一来源异常不导致全链路中断。
- 可观测到分来源健康曲线与错误原因分布。

#### S2-T2：Optuna 并行 HPO 替代串行网格
目标：新增可恢复、并行的贝叶斯优化脚本，默认替代串行 grid。

新增/改动文件：
- `scripts/optuna_liquid_hpo.py`（新增）
- `training/requirements.txt`
- `scripts/tune_liquid_strategy_grid.py`（保留为 baseline）

实施动作：
1. 引入 `optuna`，实现 `TPESampler + MedianPruner`。
2. 使用 SQLite 存储：`artifacts/hpo/optuna_liquid.db`。
3. 每 trial 实时写 JSONL：`artifacts/hpo/optuna_trials_*.jsonl`。
4. 支持 `--resume` 和 `--n-workers`。

验收命令：
- `python3 scripts/optuna_liquid_hpo.py --help`
- `python3 scripts/optuna_liquid_hpo.py --track liquid --symbols BTC,ETH,SOL --n-trials 50 --n-workers 4 --resume`

DoD：
- 中断后可续跑到同一 study。
- 在相同预算下 top-k 收敛速度优于 `scripts/tune_liquid_strategy_grid.py`。

#### S2-T3：Search/Validation/OOS 三阶段晋升
目标：把阶段化评估写入模型晋升逻辑，阻断短窗过拟合。

改动文件：
- `training/liquid_model_trainer.py`
- `training/validation.py`
- `backend/v2_router.py`
- `monitoring/model_ops_scheduler.py`

实施动作：
1. 固定三阶段窗口配置并写入训练报告。
2. 仅 Stage-3（Forward OOS）通过时允许进入 gate 候选。
3. scheduler 中 gate/rollout 读取阶段报告而不是仅看短窗均值。

验收命令：
- `pytest backend/tests/test_parity_gate.py backend/tests/test_parity_matched_fills.py -q`

DoD：
- 晋升记录中可追溯每阶段窗口与指标。
- 仅有 Search/Validation 通过但 OOS 失败时必须拒绝发布。

### Sprint 3（高价值增强，模型与信息价值闭环）

#### S3-T1：蒸馏训练路径加固与版本化
目标：让 TSMixer 蒸馏链路具备可追溯性与可回放性。

改动文件：
- `training/liquid_model_trainer.py`
- `inference/model_router.py`
- `backend/models/*`

实施动作：
1. 在 PT checkpoint 强制写入 `train_report_hash`、`feature_payload_schema_version`、`data_version`。
2. 推理加载时校验 schema/version，不匹配则拒绝并回退。
3. 训练产物增加 teacher/student 评估对照报告。

验收命令：
- `pytest backend/tests/test_model_router_core.py -q`

DoD：
- schema 不匹配的模型不会被推理路由采纳。
- 蒸馏前后指标对比可审计。

#### S3-T2：集成策略分级发布与自动回滚闭环
目标：将 `shadow -> canary -> active` 和回滚触发自动化串联。

改动文件：
- `backend/v2_router.py`
- `backend/v2_repository.py`
- `monitoring/model_ops_scheduler.py`

实施动作：
1. 固化分级阈值与晋升阶梯（10% -> 30% -> 100%）。
2. 将 drift、reject_rate、drawdown 异常并入统一回滚触发条件。
3. 回滚事件写入审计日志与风险事件表。

验收命令：
- `pytest backend/tests/test_v2_router_core.py backend/tests/test_execution_reject_realism.py -q`

DoD：
- 异常触发后可自动回滚到 `previous_model_*`。
- 审计日志可完整回放“为何晋升/为何回滚”。

#### S3-T3：信息质量到决策贡献闭环
目标：证明“新增信息”对 OOS 有正向价值，而非仅增加噪声。

改动文件：
- `training/feature_pipeline.py`
- `inference/main.py`
- `inference/explainer.py`
- `scripts/evaluate_hard_metrics.py`

实施动作：
1. 在事件与特征中纳入来源分层权重（tier-weighted aggregation）。
2. 在解释输出中加入 `source_tiers` 与缺失标记摘要。
3. 增加 A/B 特征集评估（有信息增强 vs 基线）并记录 OOS 差值。

验收命令：
- `pytest backend/tests/test_feature_decay_consistency.py backend/tests/test_ingest_event_symbol_mapping.py -q`

DoD：
- OOS 报告中可见“信息增强”相对基线的净提升或退化。
- 决策解释可追溯到具体事件来源层级。

---

## 11. 统一验收证据清单（交付时必须具备）

1. 指标证据：
- `artifacts/metrics_tests/metrics_tests.jsonl` 包含各 Sprint 的关键测试记录。

2. HPO 证据：
- `artifacts/hpo/optuna_liquid.db`
- `artifacts/hpo/optuna_trials_*.jsonl`

3. 模型证据：
- `backend/models/*` 中 manifest 与 checkpoint 字段完整。
- `model_registry` 中可追溯到 artifact 路径与训练指标。

4. 发布证据：
- rollout 状态与回滚事件可通过 API 查到完整轨迹。

5. 文档证据：
- 本文档任务勾选状态更新为完成，并附命令结果摘要。

---

## 12. 执行节奏建议（避免并行冲突）

1. 先做 Sprint 1 全部任务并冻结接口，再开始 Sprint 2。
2. Sprint 2 中 `S2-T1` 与 `S2-T2` 可并行，但 `S2-T3` 需在两者完成后执行。
3. Sprint 3 全部依赖前两轮稳定结果，建议只在门控稳定两周后启动。
