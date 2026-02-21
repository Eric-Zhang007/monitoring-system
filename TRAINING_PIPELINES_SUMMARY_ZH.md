# 训练链路总览（数据使用与特征提取）

## 1. 目的与范围

- 本文档汇总当前仓库里所有可用的训练相关链路。
- 每条链路都说明：
  - 入口脚本
  - 使用的数据表/数据文件
  - 特征如何提取
  - 标签如何定义
  - 产物与落库位置
  - 当前限制与注意事项

覆盖目录：`scripts/`、`training/`、`inference/`、`backend/`、`monitoring/`。

## 2. 总体数据流

1. 原始数据采集与审计：
   - `scripts/collect_required_data.sh`
   - `scripts/audit_required_data_readiness.py`
2. 特征层构建：
   - `scripts/build_feature_store.py` -> `feature_snapshots_main`
   - `scripts/build_text_latent_embeddings.py` -> `social_text_latent`
   - `scripts/build_market_latent_embeddings.py` -> `market_latent`
   - `scripts/merge_feature_views.py` -> `feature_matrix_main`
3. 训练层：
   - 主生产训练：`training/main.py`（VC + Liquid）
   - 多模态候选训练：`training/train_multimodal.py`
   - 多骨干对照：`training/backbone_experiments.py`
4. 评估/门禁/注册：
   - `training/eval_multimodal_oos.py`
   - `training/register_candidate_model.py`
   - `scripts/run_prod_live_backtest_batch.py`
   - `scripts/gate_training_profitability.py`
5. 持续运行：
   - `scripts/continuous_ops_loop.py`
   - `monitoring/paper_trading_daemon.py`

## 3. 训练链路清单（逐条）

## 3.1 链路 A：VC 主训练链

- 入口：
  - `training/main.py`（`TRAIN_ENABLE_VC=1`）
  - 实际训练器：`training/vc_model_trainer.py`
- 数据使用：
  - `training/feature_pipeline.py::load_vc_training_batch`
  - 来源表：`events`
  - 默认样本数：`limit=3000`
- 特征提取：
  - 5 维手工特征：
    - `event_type` 映射值（`funding/product/mna/regulatory/market`）
    - `source_tier/5`
    - `confidence_score`
    - `recency` 衰减特征
    - `recency` 原值归一化
- 标签定义：
  - 二分类标签：`event_type in {funding, mna}` 为 1，其它为 0
- 模型与训练：
  - 先算闭式线性 teacher，再训练 `TinyVCModel`（MLP）蒸馏
- 产物：
  - `backend/models/vc_survival_baseline_v2.json`
  - `backend/models/vc_survival_ttm_v2.pt`
  - `backend/models/vc_train_manifest_v2.json`
  - `model_registry` 写入 VC 记录
- 当前限制：
  - 样本来自事件类型映射，特征仍较浅，偏“事件生存/打分”而非复杂时序学习。

## 3.2 链路 B：Liquid 主生产训练链（当前核心）

- 入口：
  - `training/main.py`（`TRAIN_ENABLE_LIQUID=1`）
  - 实际训练器：`training/liquid_model_trainer.py`
- 数据使用：
  - `training/feature_pipeline.py::load_liquid_training_batch`
  - 主要来源：
    - `market_bars`（主）
    - `prices`（fallback）
    - `orderbook_l2`
    - `funding_rates`
    - `onchain_signals`（当前仅 `netflow/exchange_netflow/net_inflow`）
    - `events` + `event_links` + `entities`
    - `social_text_latent.agg_features`
  - 训练样本窗口：
    - 每 symbol 默认 `limit=4000`
    - 历史回看 `history_len=96`
- 特征提取（按样本时间 `t`）：
  - 价格/波动特征：
    - `ret_1/3/12/48/96/288`
    - `vol_3/12/48/96/288`
    - `ret_accel_*`、`vol_term_*`
  - 成交量特征：
    - `log_volume`、`vol_z`、`volume_impact`
  - 外生数值特征（as-of）：
    - `orderbook_imbalance`（最近可用）
    - `funding_rate`（最近可用）
    - `onchain_norm=tanh(flow/1e6)`（最近可用）
    - 缺失标记：`orderbook_missing_flag/funding_missing_flag/onchain_missing_flag`
  - 事件/社媒特征：
    - 从 `events` 提取 24h 内事件，严格 no-lookahead（仅 `timestamp<=t`）
    - 使用 `available_at` 优先对齐，降泄露风险
    - 生成 `event_decay/source_confidence/event_velocity/source_diversity/...`
    - 再与 `social_text_latent.agg_features` 的 6h 窗口统计做 blend（`SOCIAL_AGG_BLEND_ALPHA`）
- 标签定义：
  - 主标签：`fwd_ret_1h - est_cost`
  - 附加标签：`fwd_ret_4h - 2*est_cost`
  - `est_cost` 由手续费+滑点+盘口冲击近似构造
- 模型与训练：
  - teacher：LightGBM（缺依赖时 Ridge fallback）
  - student：TSMixer（MSE + 蒸馏损失）
  - 支持文本模态 dropout（按特征名自动识别 text/event 维度）
  - 支持 DDP（`DistributedDataParallel`）
- 评估与验证：
  - 内置 Walk-forward + Purged KFold（`training/validation.py`）
  - 输出成本后 OOS 指标（IC/Hit/Turnover/PnL/DD）
- 产物：
  - `backend/models/liquid_{symbol}_lgbm_baseline_v2.json`
  - `backend/models/liquid_{symbol}_tsmixer_v2.pt`
  - `backend/models/liquid_{symbol}_train_manifest_v2.json`
  - `backend/models/liquid_{symbol}_tsmixer_v2.manifest.json`
  - `feature_snapshots` 落训练特征快照
  - `model_registry` 写入 tabular + tsmixer
- 当前限制：
  - 默认每 symbol 只取 4000 条样本，不是全历史全量训练。
  - 新增的 derivatives 指标尚未进入该链路的训练特征集合。

## 3.3 链路 C：多模态候选训练链（feature_matrix_main）

- 入口：`training/train_multimodal.py`
- 数据使用：
  - 来源表：`feature_matrix_main`
  - 时间/资产过滤：`--start --end --symbols`
  - 样本构造：按 symbol 排序后，用第 `i+horizon` 行的 `ret_1` 作为目标
- 特征提取：
  - 输入向量：`inference/liquid_feature_contract.py::LIQUID_FEATURE_KEYS`
  - 训练前标准化：`x_mean/x_std`
  - 自动识别文本相关维度（`social_*`, `event_*`, `source_*`, `sentiment_*` 等）
  - 文本维度做 dropout（训练阶段）
- 标签定义：
  - `future_ret_1_step_horizon_steps`
- 融合模式：
  - `single_ridge`：单路 Ridge
  - `residual_gate`：
    - base（非文本）预测
    - text 预测残差
    - gate(sigmoid(text_activity)) 控制 text 残差贡献
- 产物：
  - `artifacts/models/multimodal_candidate.json`
- 当前限制：
  - 依赖 `feature_matrix_main` 完整性，若行数不足会直接退出。

## 3.4 链路 D：多模态 OOS 与消融评估链

- 入口：`training/eval_multimodal_oos.py`
- 数据使用：
  - 来源表：`feature_matrix_main`
  - 与链路 C 相同的目标定义（future row `ret_1`）
- 特征与消融：
  - `full`：全特征
  - `no_text`：文本相关维置零
  - `no_macro`：宏观相关维（funding/onchain 等）置零
  - `event_window`：仅保留事件强度高于阈值的样本
- 验证协议：
  - 统一 WF + Purged KFold（来自 `training/validation.py`）
  - 输出误差指标 + 成本后 OOS 指标
- 产物：
  - `artifacts/models/multimodal_eval.json`
- 当前限制：
  - 无已知阻断缺陷；Phase 0 已补齐 smoke 测试覆盖该链路，防回归。

## 3.5 链路 E：候选注册与门禁链

- 入口：`training/register_candidate_model.py`
- 数据使用：
  - 输入工件：
    - `multimodal_candidate.json`
    - `multimodal_eval.json`
    - `backbone_suite_latest.json`（可选）
- 特征/指标处理：
  - 聚合 ablation 摘要（`delta_mse_no_text_vs_full` 等）
  - 检查 OOS 指标阈值、ready backbones、required backbones
- 产物：
  - 追加写入 `artifacts/models/candidate_registry.jsonl`
  - DB 表 `candidate_models` 插入记录
- 当前限制：
  - 是否强制阻断取决于 `--enforce-gates` / 环境变量。

## 3.6 链路 F：多骨干统一对照链

- 入口：`training/backbone_experiments.py`
- 数据使用：
  - 来源表：`feature_matrix_main`
  - 序列样本构造：`lookback_steps` 窗口 + `horizon_steps` 目标
  - 可用 `max_samples` 下采样
- 特征提取：
  - 输入 shape：`[batch, lookback_steps, feature_dim]`
  - 每个 fold 只用训练集统计量做标准化
- 模型：
  - `ridge`（flatten 后线性）
  - `itransformer`（lite）
  - `patchtst`（lite）
  - `tft`（lite）
- 产物：
  - `artifacts/experiments/backbone_suite_latest.json`
  - 可由 `training/main.py` 的 `TRAIN_ENABLE_BACKBONE_EXPERIMENTS=1` 挂到主训练定时流程
- 当前限制：
  - 需要 `torch` 才能跑神经骨干。
  - 若环境缺少 `torch`，`tft/itransformer/patchtst` 会被标记为 `torch_missing`（显式降级），仅 `ridge` 可运行。

## 3.7 链路 G：服务器双卡训练编排链

- 入口：
  - `scripts/launch_dual_a100_training.sh`
  - `scripts/train_gpu_stage2.py`
- 数据使用：
  - 不直接提特征，调用 `training/main.py` 触发链路 A/B/F
- 特征提取：
  - 复用链路 B（Liquid）与链路 A（VC）的特征提取
- 核心能力：
  - GPU 数量/型号/NVLINK 预检
  - 自动选择 `torchrun --nproc_per_node`
  - `TRAIN_NPROC_PER_NODE`、`LIQUID_SYMBOL_DDP` 等环境注入
- 产物：
  - `artifacts/gpu_stage2/*.json` 运行记录

## 3.8 链路 H：持续训练与模拟盘联动链

- 入口：
  - `scripts/continuous_ops_loop.py`
  - `monitoring/paper_trading_daemon.py`
- 数据使用：
  - 通过命令串联训练、回测、门禁、审计
  - 模拟盘通过 API 取信号/预测并下 paper 订单
- 特征提取：
  - 训练特征提取复用链路 B/C/F
  - 模拟盘信号来自线上推理特征（`feature_snapshots` + `ModelRouter`）
- 联动逻辑：
  - 训练成功后才开模拟盘（可配置）
  - 收益门禁失败时触发微调命令（可配置）
- 产物：
  - `artifacts/ops/continuous_runtime_state.json`
  - `artifacts/ops/continuous_runtime_history.jsonl`
  - `artifacts/paper/*`

## 4. 特征层构建链路（训练前）

## 4.1 `build_feature_store.py`（手工主特征）

- 输入：`market_bars`（按 symbol/timeframe/time window）
- 处理：
  - 生成 `ret/vol/volume_impact` 等手工特征
  - 写入 `feature_snapshots_main`
- 输出表：`feature_snapshots_main`

## 4.2 `build_text_latent_embeddings.py`（社媒 latent + 统计）

- 输入：`events` 中社媒事件（按 `social_platform` 过滤）
- 处理：
  - 5m 桶聚合（symbol, bucket_end）
  - post/comment 文本哈希向量
  - engagement 加权集合 pooling
  - 统计特征 `agg_features`（post/comment count、情绪、作者数、engagement 等）
- 输出表：`social_text_latent`

## 4.3 `build_market_latent_embeddings.py`（市场 latent）

- 输入：`market_bars`
- 处理：
  - 时间窗口内抽取市场序列 latent 向量
- 输出表：`market_latent`

## 4.4 `merge_feature_views.py`（最终训练矩阵）

- 输入：
  - `feature_snapshots_main`（manual）
  - `market_latent`
  - `social_text_latent`（latent + agg_features）
- 处理：
  - as-of 对齐 `market_latent`（latest_before）
  - social 6h 窗口聚合（均值向量 + 统计 blend）
  - 合成 `LIQUID_FULL_FEATURE_KEYS`
- 输出表：`feature_matrix_main`

## 5. 当前已有数据（审计口径）

- 数据来源依据：
  - `artifacts/audit/required_data_readiness_latest.json`（生成时间：`2026-02-19T23:53:17Z`）
  - `artifacts/audit/full_history_latest.json`（窗口截至：`2026-02-19T23:52:21Z`）

- 当前资产与时间粒度：
  - 资产池：`BTC, ETH, SOL, BNB, XRP, ADA, DOGE, TRX, AVAX, LINK`
  - 主周期：`5m`
  - 次周期：`1h`

- 当前已存在的数据表（训练相关）：
  - 已存在：`market_bars`, `orderbook_l2`, `funding_rates`, `onchain_signals`, `events`, `event_links`, `entities`
  - 不存在（原始社媒明细表）：`social_posts_raw`, `social_comments_raw`
  - 说明：社媒相关训练信号当前主要来自 `events.payload` 与 `social_text_latent` 聚合表。

- 420 天训练准备窗口（`lookback_start=2024-12-26`）覆盖情况：
  - `market_bars(5m)` 平均覆盖：`0.99975199`
  - `market_bars(1h)` 平均覆盖：`0.99841286`
  - `orderbook_l2` 平均覆盖：`0.99975199`
  - `funding_rates` 平均覆盖：`0.99994213`
  - `onchain_primary_metric(net_inflow)` 平均覆盖：`0.99982639`
  - 事件与社媒统计：
    - `linked_events_total=1920`（均值每币 `192.0`）
    - `posts_total=480`
    - `comments_total=24110`
    - `comment_post_ratio=50.22916667`
  - 就绪门禁：`gates.ready=true`，`missing_data_kinds=[]`

- 衍生品增强指标（近 30 天，5m 对齐）：
  - 已覆盖指标：
    - `long_short_ratio_global_accounts`
    - `long_short_ratio_top_accounts`
    - `long_short_ratio_top_positions`
    - `taker_buy_sell_ratio`
    - `basis_rate`
    - `annualized_basis_rate`
  - 审计覆盖：上述指标平均覆盖约 `0.99976855`（10 币一致）。

- 2018-now 全窗口（5m）覆盖情况：
  - 窗口：`2018-01-01` 到 `2026-02-19`
  - 每币理论桶数：`855,935`
  - 全体理论桶数（10 币）：`8,559,350`
  - 实际覆盖（`full_history_latest`）：
    - `market_bars`：`5,888,759 / 8,559,350`（`0.68799138`）
    - `orderbook_l2`：`5,888,759 / 8,559,350`（`0.68799138`）
    - `funding_rates`：`6,266,426 / 8,559,350`（`0.7321147`）
    - `onchain_signals`：`5,888,909 / 8,559,350`（`0.6880089`）
    - `social_posts`：`405 / 855,935`（`0.00047317`）
    - `social_comments`：`405 / 855,935`（`0.00047317`）
  - 全窗口结论：
    - `history_window_complete=false`
    - `audit_ready_for_next_phase=false`
    - 即：短中期训练窗口已就绪，但 2018-now 全窗口仍未补齐。

## 6. 当前“全量数据使用”状态说明

- 链路 B（Liquid 主生产）：
  - 已升级为“窗口优先”：`LIQUID_TRAIN_MODE=production` 时默认不使用硬 `limit=4000`，通过 `LIQUID_TRAIN_LOOKBACK_DAYS` 或 `--liquid-start/--liquid-end` 控窗。
  - `LIQUID_TRAIN_MODE=fast` 才使用 `liquid_limit/liquid_max_samples` 进行快速迭代。
  - derivatives 指标已接入主训练特征，并保留 missing_flag 与 as-of 对齐。
- 链路 C/D/F（基于 `feature_matrix_main`）：
  - 可以按 `--start --end` 使用全窗口，受表本身覆盖与 `max_samples` 参数影响。
- 链路 A（VC）：
  - 默认 `limit=3000`，不是全事件历史。

## 7. 当前后续重点（按影响排序）

1. 在服务器完整数据环境下重跑 multi-horizon OOS + paper，校准 `SIGNAL_SCORE_ENTRY_BY_HORIZON` 与 `SIGNAL_CONFIDENCE_MIN_BY_HORIZON`。
2. 持续积累 `backtest_runs` 后启用严格门禁：`min_improved_wf_windows>=2` 且成本后优于 Phase 0 baseline。
3. 执行 `symbol+horizon` 级 candidate->active->rollback 运维演练，固化值班流程。
4. 持续监控 `/api/v2/monitor/horizon-performance`、`prediction-drift`、`confidence-calibration`、`paper-pnl-buckets`。
5. 每次改动后运行 `scripts/run_multi_horizon_upgrade_bundle.py`，统一落 `artifacts/upgrade/multi_horizon_upgrade_bundle_latest.json`。

---

更新时间：`2026-02-20`  
维护人：`Codex`
