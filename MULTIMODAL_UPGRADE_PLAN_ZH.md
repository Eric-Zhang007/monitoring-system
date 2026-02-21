# 多模态升级重构计划与进度追踪（5m K线 + 宏观 + 社媒）

## 0. 文档约定

- `目标`：将当前仓库升级为“时间严格对齐、文本可降噪、融合可控、可回测可上线”的多模态量化系统。
- `范围`：`collector/`、`scripts/`、`training/`、`inference/`、`backend/`、`monitoring/`。
- `状态枚举`：
  - `TODO`：未开始
  - `IN_PROGRESS`：进行中
  - `BLOCKED`：被阻塞
  - `DONE`：已完成
- `维护规则`（必须执行）：
  - 每次开始一个子任务前，先更新“任务状态表”中的负责人、起止时间、状态。
  - 每次完成一个子任务后，补“变更记录”和“下一步”。
  - 每天结束前更新“剩余风险”和“明日优先级”。

## 1. 当前仓库基线（2026-02-19）

### 1.1 已有能力

- 采集与入库：
  - 社媒 ingest 路由：`backend/routers/multimodal.py`
  - 社媒到事件转换：`backend/services/multimodal_service.py`
- 特征与融合：
  - 手工特征+事件特征：`training/feature_pipeline.py`
  - 文本 latent 构建（hash 向量）：`scripts/build_text_latent_embeddings.py`
  - 多视图合并到 `feature_matrix_main`：`scripts/merge_feature_views.py`
- 训练与评估：
  - 主线训练器（TSMixer + LGB/Ridge fallback）：`training/liquid_model_trainer.py`
  - 多模态基线训练（Ridge）：`training/train_multimodal.py`
  - OOS 评估脚本：`training/eval_multimodal_oos.py`
- 防泄露：
  - as-of 对齐验证：`scripts/validate_asof_alignment.py`
  - no-leakage 验证：`scripts/validate_no_leakage.py`

### 1.2 已识别核心缺口

- 文本表征仍偏弱：当前为 hash 向量，缺少“帖子+评论集合聚合 + 可学习降维 + 门控”。
- 融合方式偏静态：`merge_feature_views.py` 当前主要是均值/拼接，缺少“残差 + gate”。
- 多模型对照不足：缺少 iTransformer/PatchTST/TFT/TSMixer 的统一实验协议。
- 交易层目标不足：评估指标以回归误差为主，交易成本后稳健性门禁不完整。

## 2. 总体目标与验收门禁

## 2.1 最终目标（升级完成定义）

- 文本模态经过严格时间对齐与降噪，支持在高噪场景自动“降权/忽略”。
- 数值主干支持至少两条生产可选路线（例如 `TSMixer` 与 `iTransformer/TFT`）。
- 训练、评估、注册、上线、监控形成闭环，并可回溯到 lineage 与数据版本。

## 2.2 硬门禁（未通过不得晋级）

- `Leakage Gate`：
  - `scripts/validate_asof_alignment.py` 通过
  - `scripts/validate_no_leakage.py` 通过（`violating_runs=0`）
- `Data Quality Gate`：
  - 历史完整性审计通过（沿用当前全窗口审计脚本）
  - 社媒帖子/评论吞吐满足最低覆盖阈值
- `Model Gate`：
  - 至少 2 个 walk-forward 周期在成本后优于当前生产基线
  - 稳定性（回撤、换手、拒单率）不过线
- `Ops Gate`：
  - `/api/v2/monitor/*` 增加多模态健康项并稳定暴露
  - 回滚路径可用（candidate -> active -> rollback）

## 3. 分阶段升级路线图

## 3.1 Phase A：基线冻结与实验框架统一

- 目标：建立可重复对比的统一实验协议，冻结当前基线结果。
- 关键输出：
  - 基线模型清单与固定参数（TSMixer/LGB/Ridge）
  - 统一数据切分协议（walk-forward + purged kfold）
  - 统一评估报告模板（误差 + 方向 + 成本后指标）
- 主要改动点：
  - `training/eval_multimodal_oos.py`
  - `scripts/run_2025_2026_validation_bundle.sh`
  - `artifacts/` 下新增实验记录目录

## 3.2 Phase B：多模态对齐与文本降噪（先稳后强）

- 目标：把“帖子+评论”从原始噪声转为低维、可控、可解释输入。
- 关键输出：
  - 5m 桶级对齐特征（热度/情绪/扩散/事件词）
  - 文本集合聚合（post/comment 分塔 -> pooling -> 压缩到 16~64 维）
  - 模态 dropout（训练中随机失活文本模态）
- 主要改动点：
  - `scripts/build_text_latent_embeddings.py`
  - `training/feature_pipeline.py`
  - `scripts/merge_feature_views.py`
  - `inference/liquid_feature_contract.py`（新增/下线特征时同步版本）

## 3.3 Phase C：融合架构升级（门控/残差优先）

- 目标：让模型学会“何时相信文本、何时忽略文本”。
- 关键输出：
  - 主干预测：`y_base = f(price, macro, microstructure)`
  - 文本残差：`delta = f_text(text_features)`
  - 门控融合：`y = y_base + gate * delta`
- 主要改动点：
  - `training/train_multimodal.py`（新增 gate residual 训练范式）
  - `training/liquid_model_trainer.py`（接入可选融合头）
  - `inference/model_router.py` / `backend/liquid_model_service.py`（推理兼容）

## 3.4 Phase D：骨干模型对照（iTransformer / PatchTST / TFT / TSMixer）

- 目标：在统一协议下比较收益，不做“单模型信仰”。
- 关键输出：
  - 至少 4 条可复现实验（每条含配置、种子、结果、工件）
  - 消融实验（去文本、去宏观、仅事件爆发窗口）
  - 最终候选排序与上线建议
- 主要改动点：
  - `training/main.py`（统一模型入口）
  - `training/validation.py`（扩展成本后评估）
  - `training/register_candidate_model.py`（记录融合/文本版本元数据）

## 3.5 Phase E：上线闭环与监控加固

- 目标：将新多模态方案接入生产控制面，并可观测、可回滚。
- 关键输出：
  - 新模型工件契约与 schema 版本切换流程
  - 新监控指标（文本覆盖、gate 开度、文本贡献度、预测漂移）
  - 运营手册更新（上线、降级、回滚）
- 主要改动点：
  - `backend/main.py` / `backend/v2_router.py` / `backend/metrics.py`
  - `monitoring/model_ops_scheduler.py` / `monitoring/paper_trading_daemon.py`
  - `README.md` / `TRACKING.md`

## 4. 详细任务分解（持续维护）

| ID | 阶段 | 任务 | 关键文件 | 验收标准 | 状态 | 计划开始 | 计划完成 | 实际完成 |
|---|---|---|---|---|---|---|---|---|
| A-1 | Phase A | 冻结当前生产基线配置与工件清单 | `training/liquid_model_trainer.py`, `backend/models/` | 产出基线清单文档 + 校验脚本 | DONE | 2026-02-19 | 2026-02-20 | 2026-02-19 |
| A-2 | Phase A | 统一 walk-forward + purged kfold 协议 | `training/validation.py`, `training/eval_multimodal_oos.py` | 全模型走同一切分配置 | DONE | 2026-02-20 | 2026-02-21 | 2026-02-19 |
| A-3 | Phase A | 标准化评估报告模板（含成本后） | `scripts/evaluate_hard_metrics.py`, `artifacts/` | 每次实验自动落 JSON 报告 | DONE | 2026-02-20 | 2026-02-22 | 2026-02-19 |
| B-1 | Phase B | 实现 5m 桶级文本集合聚合器（post/comment） | `scripts/build_text_latent_embeddings.py` | 聚合输出可重复、无未来信息 | DONE | 2026-02-21 | 2026-02-23 | 2026-02-19 |
| B-2 | Phase B | 补充稳健文本统计特征（热度/情绪/突发） | `training/feature_pipeline.py` | 新特征进入 `feature_snapshots_main` | DONE | 2026-02-22 | 2026-02-24 | 2026-02-19 |
| B-3 | Phase B | 文本向量压缩到 16~64 维并做版本化 | `scripts/merge_feature_views.py`, `inference/liquid_feature_contract.py` | 维度受控，兼容线上 schema | DONE | 2026-02-23 | 2026-02-25 | 2026-02-19 |
| B-4 | Phase B | 模态 dropout 训练支持 | `training/train_multimodal.py` | 文本缺失场景性能不崩 | DONE | 2026-02-24 | 2026-02-25 | 2026-02-19 |
| C-1 | Phase C | 落地 residual + gate 融合训练 | `training/train_multimodal.py` | 可输出 gate 与残差贡献 | DONE | 2026-02-25 | 2026-02-27 | 2026-02-19 |
| C-2 | Phase C | 将融合推理接入线上模型服务 | `inference/model_router.py`, `backend/liquid_model_service.py` | 推理端支持新工件类型 | DONE | 2026-02-26 | 2026-02-28 | 2026-02-19 |
| C-3 | Phase C | 新增融合相关单测 | `backend/tests/`, `training/` | 覆盖 gate 边界与回退路径 | DONE | 2026-02-26 | 2026-02-28 | 2026-02-19 |
| D-1 | Phase D | 接入 iTransformer 实验分支 | `training/main.py`, `training/` | 生成可复现实验结果 | IN_PROGRESS | 2026-02-28 | 2026-03-02 | - |
| D-2 | Phase D | 接入 PatchTST 实验分支 | `training/main.py`, `training/` | 生成可复现实验结果 | IN_PROGRESS | 2026-03-01 | 2026-03-03 | - |
| D-3 | Phase D | 接入 TFT 实验分支（特征门控对照） | `training/main.py`, `training/` | 生成可复现实验结果 | IN_PROGRESS | 2026-03-02 | 2026-03-04 | - |
| D-4 | Phase D | 统一消融实验（去文本/去宏观/事件窗口） | `training/eval_multimodal_oos.py` | 完成 3 组消融报告 | IN_PROGRESS | 2026-03-03 | 2026-03-05 | - |
| E-1 | Phase E | 监控指标扩展（文本覆盖、gate 开度） | `backend/metrics.py`, `backend/main.py` | `/metrics` 可采集新指标 | DONE | 2026-03-05 | 2026-03-06 | 2026-02-19 |
| E-2 | Phase E | ModelOps 门禁接入新评估项 | `monitoring/model_ops_scheduler.py`, `scripts/validate_no_leakage.py` | 自动阻断不达标候选 | DONE | 2026-03-05 | 2026-03-07 | 2026-02-19 |
| E-3 | Phase E | 发布与回滚演练 | `scripts/server_up.sh`, `scripts/server_readiness.sh` | 演练记录通过 | DONE | 2026-03-06 | 2026-03-07 | 2026-02-19 |
| E-4 | Phase E | 更新部署与运行文档 | `README.md`, `REPOSITORY_ARCHITECTURE_GUIDE_ZH.md` | 文档与代码一致 | DONE | 2026-03-06 | 2026-03-07 | 2026-02-19 |
| F-1 | Phase F | 数据窗口与模态完整性盘点（训练视角） | `scripts/audit_full_history_completeness.py`, `scripts/audit_training_data_completeness.py` | 输出缺口报告 + 分模态优先级 | DONE | 2026-02-19 | 2026-02-20 | 2026-02-20 |
| F-2 | Phase F | 衍生品信号采集（funding + OI proxy） | `scripts/ingest_binance_aux_signals.py` | `funding_rates/onchain_signals` 非零覆盖 | DONE | 2026-02-19 | 2026-02-21 | 2026-02-20 |
| F-3 | Phase F | 订单簿代理回填（用于冷启动） | `scripts/backfill_orderbook_from_market_bars.py` | `orderbook_l2` 冷启动窗口可用 | DONE | 2026-02-19 | 2026-02-21 | 2026-02-20 |
| F-4 | Phase F | 缺口修复编排脚本（可直上服务器） | `scripts/remediate_liquid_data_gaps.sh` | 一键执行并输出审计报告 | DONE | 2026-02-19 | 2026-02-21 | 2026-02-20 |
| F-5 | Phase F | 2026+ 额外数据路线评估与实施清单 | `MULTIMODAL_UPGRADE_PLAN_ZH.md`, `README.md` | 给出可执行的数据扩展路线图 | DONE | 2026-02-19 | 2026-02-20 | 2026-02-20 |
| F-6 | Phase F | 额外模态采集实现（优先复用现有脚本） | `scripts/orchestrate_event_social_backfill.py`, `scripts/build_multisource_events_2025.py` | 新增数据源不破坏既有回填链路 | DONE | 2026-02-19 | 2026-02-22 | 2026-02-20 |
| F-7 | Phase F | 本地采集 -> 服务器离线导入链路 | `scripts/build_offline_data_bundle.sh`, `scripts/import_offline_data_bundle.sh`, `scripts/server_import_offline_bundle.sh` | 服务器无外网时仍可完成数据导入与训练 | DONE | 2026-02-19 | 2026-02-22 | 2026-02-20 |
| F-8 | Phase F | 衍生品增强数据采集（长短仓比/主动买卖比/基差） | `scripts/ingest_binance_derivatives_signals.py` | `onchain_signals` 新增 derivatives metric 覆盖并可审计 | DONE | 2026-02-19 | 2026-02-22 | 2026-02-20 |
| F-9 | Phase F | 必要数据一键采集与就绪审计总入口 | `scripts/collect_required_data.sh`, `scripts/audit_required_data_readiness.py` | 可一键执行采集->复审->离线包，全过程落产物 | DONE | 2026-02-19 | 2026-02-22 | 2026-02-20 |
| G-1 | Phase G | 训练收益门禁脚本与调度接入 | `scripts/gate_training_profitability.py`, `scripts/continuous_ops_loop.py` | 可自动判定“训练是否有收益”，并可触发后续动作 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |
| G-2 | Phase G | 双卡 A100 NVLINK 训练启动器 | `scripts/launch_dual_a100_training.sh`, `scripts/train_gpu_stage2.py` | 启动前完成双卡/NVLINK 预检并一键发车 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |
| G-3 | Phase G | 首训成功后模拟盘 + 门禁失败自动微调 | `scripts/continuous_ops_loop.py`, `monitoring/paper_trading_daemon.py` | 首训前不下模拟单；收益门禁失败可自动微调 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |

## 5. 里程碑追踪板（高层）

| 里程碑 | 目标日期 | 当前状态 | 完成度 |
|---|---|---|---|
| M1：统一实验协议与基线冻结 | 2026-02-22 | DONE | 100% |
| M2：文本对齐降噪 v2 | 2026-02-25 | DONE | 100% |
| M3：门控残差融合可训练可推理 | 2026-02-28 | DONE | 100% |
| M4：多骨干对照与消融闭环 | 2026-03-05 | IN_PROGRESS | 80% |
| M5：上线监控与回滚闭环 | 2026-03-07 | DONE | 100% |

## 6. 风险与阻塞管理（持续维护）

| 风险ID | 描述 | 影响 | 缓解动作 | 当前状态 |
|---|---|---|---|---|
| R-1 | 文本特征维度过大导致过拟合/推理慢 | 高 | 强制降维 16~64；加模态 dropout；门控融合 | OPEN |
| R-2 | 时间对齐误差引入隐性泄露 | 高 | 发布时刻对齐；固定 shift；每日 leakage gate | OPEN |
| R-3 | 多模型并行实验导致配置漂移 | 中 | 单一配置模板 + 固定 seed + 工件签名 | OPEN |
| R-4 | 成本后收益不稳定 | 高 | 纳入交易成本与容量约束，按窗口做稳健性筛选 | OPEN |
| R-5 | 上线后观测不足无法快速回滚 | 高 | 增加 gate 开度/文本贡献监控 + 回滚演练 | OPEN |
| R-6 | baseline ensemble 清单引用了缺失的 `.pt` 组件 | 中 | 已确认延后到服务器训练阶段补齐；当前本地阶段不阻塞 A/B/C 设计与实现 | DEFERRED |
| R-7 | 系统 `python3` 缺 `numpy/psycopg2`，运行脚本需统一使用 `.venv/bin/python` | 中 | 本地测试统一切到 `.venv`；服务器按 requirements 固化依赖 | OPEN |
| R-8 | 本地 `.venv` 缺 `torch`，iTransformer/PatchTST/TFT 实验仅能在服务器运行 | 中 | 本地先完成脚本/协议与单测，服务器环境按 `requirements-train.txt` 执行实验 | OPEN |
| R-9 | Binance `openInterestHist` 存在历史窗口限制（过旧时间无效） | 中 | 脚本内置窗口钳制 + market_bars 派生 fallback，避免 onchain 特征断层 | OPEN |
| R-10 | 服务器出网受限导致采集任务无法直接执行 | 高 | 固化“本地采集打包 + 服务器离线导入”流程；支持 `scp` 或 GitHub 分发 | OPEN |
| R-11 | Binance futures/data 衍生品指标普遍存在短历史保留（约 30 天） | 中 | 增强“近期覆盖门禁”而非强制全窗口；必要时引入付费历史源 | OPEN |
| R-12 | 本地环境无双 A100/NVLINK，双卡链路只能做脚本级预演 | 中 | 新增 GPU/NVLINK 预检脚本，服务器上线前做一次真实 dry-run + 实跑校验 | OPEN |

## 7. 变更记录（持续追加）

| 日期 | 变更人 | 变更内容 | 影响里程碑 |
|---|---|---|---|
| 2026-02-19 | Codex | 新建本计划文档；完成当前仓库基线盘点；建立任务与风险追踪框架。 | M1 |
| 2026-02-19 | Codex | 完成 A-1：新增 `scripts/freeze_liquid_baseline.py`，产出 `baseline_snapshots/liquid_baseline_freeze_20260219T000000Z.json` 与冻结说明文档。 | M1 |
| 2026-02-19 | User+Codex | 决策确认：所有 `.pt` 组件统一延后到服务器训练阶段产出；当前本地重构阶段允许缺失并继续推进。 | M1 |
| 2026-02-19 | Codex | 完成 A-2：在 `training/validation.py` 新增统一验证协议加载器，`training/liquid_model_trainer.py` 与 `training/eval_multimodal_oos.py` 统一复用同一套 WF/PKF 配置与阈值。 | M1 |
| 2026-02-19 | Codex | 完成 A-3：`scripts/evaluate_hard_metrics.py` 增加标准报告模板并默认写入 `artifacts/eval_reports/hard_metrics/*.json`，同时保留 stdout 兼容。 | M1 |
| 2026-02-19 | Codex | 完成 B-1：`scripts/build_text_latent_embeddings.py` 升级为 5m 桶级 symbol 集合聚合（post/comment pooling），并新增 `agg_features` 聚合字段与窗口重建逻辑。 | M2 |
| 2026-02-19 | Codex | 推进 B-2（第一版）：`scripts/merge_feature_views.py` 接入 `agg_features`，将热度/情绪/评论占比/事件密度等稳健文本统计融合进最终训练矩阵。 | M2 |
| 2026-02-19 | Codex | 推进 B-3（第一版）：`scripts/build_text_latent_embeddings.py` 支持可配置压缩维度（默认 32+32），`scripts/merge_feature_views.py` 按契约目标维度对齐 social latent。 | M2 |
| 2026-02-19 | Codex | 刷新基线快照：`baseline_snapshots/liquid_baseline_freeze_20260219T020000Z.json`，纳入 `SOCIAL_POST_LATENT_DIM/SOCIAL_COMMENT_LATENT_DIM/SOCIAL_AGG_BLEND_ALPHA` 配置签名。 | M2 |
| 2026-02-19 | Codex | 推进 B-4（第一版）：`training/train_multimodal.py` 增加文本模态 dropout（`--text-dropout-prob`）与文本特征索引自动识别。 | M2 |
| 2026-02-19 | Codex | 完成 B-2：`training/feature_pipeline.py` 下沉接入 `social_text_latent.agg_features`，并与事件画像按 `SOCIAL_AGG_BLEND_ALPHA` 融合。 | M2 |
| 2026-02-19 | Codex | 完成 B-3：`inference/liquid_feature_contract.py` 增加 schema->latent 维度映射（支持 `main_64` 等版本）；压缩维度链路已可配置。 | M2 |
| 2026-02-19 | Codex | 推进 B-4（第二版）：`training/liquid_model_trainer.py` 接入文本模态 dropout 到 LGB 与 TSMixer 训练路径，并写入 train manifest。 | M2 |
| 2026-02-19 | Codex | 刷新基线快照：`baseline_snapshots/liquid_baseline_freeze_20260219T030000Z.json`，纳入 `LIQUID_TEXT_DROPOUT_PROB/MULTIMODAL_TEXT_DROPOUT_PROB` 配置签名。 | M2 |
| 2026-02-19 | Codex | 完成 B-4：新增 `backend/tests/test_train_multimodal_dropout.py` 覆盖文本特征识别与模态 dropout 的可复现行为，`.venv` 定向测试通过。 | M2 |
| 2026-02-19 | Codex | 完成 C-1：确认 `training/train_multimodal.py` 的 `residual_gate` 训练范式可输出 `base/text/gate` 关键参数并落工件。 | M3 |
| 2026-02-19 | Codex | 推进 C-2/C-3：`inference/model_router.py` 新增 `residual_gate` tabular 工件推理兼容与 stack 字段暴露；新增 `backend/tests/test_model_router_residual_fusion.py` 并通过。 | M3 |
| 2026-02-19 | Codex | 完成 C-2：`backend/liquid_model_service.py` 支持缺失 `.pt` 时按配置退化到 tabular（`LIQUID_REQUIRE_NN_ARTIFACT`），并在输出中暴露 `degraded`/`neural_artifact_present`。 | M3 |
| 2026-02-19 | Codex | 完成 C-3：新增 `backend/tests/test_liquid_model_service_artifact_fallback.py`，覆盖 gate/融合回退关键路径；定向回归 `13 passed`。 | M3 |
| 2026-02-19 | Codex | 刷新基线快照：`baseline_snapshots/liquid_baseline_freeze_20260219T040000Z.json`，纳入 `MULTIMODAL_FUSION_MODE` 配置签名并更新 latest。 | M3 |
| 2026-02-19 | Codex | 启动 D-1/D-2：新增 `training/backbone_experiments.py`，统一接入 `ridge/itransformer/patchtst` 多骨干实验入口并复用同一 WF/PKF 验证协议。 | M4 |
| 2026-02-19 | Codex | 推进 D-1/D-2：`training/main.py` 增加 `TRAIN_ENABLE_BACKBONE_EXPERIMENTS` 开关；`README.md` 补充统一多骨干实验命令与运行说明。 | M4 |
| 2026-02-19 | Codex | 新增 `backend/tests/test_backbone_experiments_dataset.py` 覆盖序列样本构建与可重复下采样逻辑；定向测试通过。 | M4 |
| 2026-02-19 | Codex | 推进 D 阶段注册链路：`training/register_candidate_model.py` 新增 `--backbone-report`，将 ready backbones / torch 可用性写入 candidate registry payload。 | M4 |
| 2026-02-19 | Codex | 启动 D-4：`training/eval_multimodal_oos.py` 增加 `full/no_text/no_macro/event_window` 消融评估与统一输出 `ablation_results`，兼容原主指标字段。 | M4 |
| 2026-02-19 | Codex | 新增 `backend/tests/test_eval_multimodal_ablations.py` 覆盖消融分组与事件窗口筛选逻辑；定向测试通过。 | M4 |
| 2026-02-19 | Codex | `README.md` 补充消融评估命令；并修复 `training/eval_multimodal_oos.py` 直跑导入路径问题。 | M4 |
| 2026-02-19 | Codex | 推进 D-4 注册链路：`training/register_candidate_model.py` 增加 `ablation_summary` 聚合，写入候选注册 payload，并新增 `backend/tests/test_register_candidate_ablation_summary.py`。 | M4 |
| 2026-02-19 | Codex | 面向服务器执行补齐：新增 `scripts/run_phase_d_multimodal_bundle.sh` 一键流水线脚本（训练/多骨干/消融/注册），并新增注册门禁参数化配置与返回码。 | M4 |
| 2026-02-19 | Codex | 配置模板更新：`.env.example` 新增 Phase-D 与 Candidate Gate 参数；`README.md` 补充一键脚本与配置说明。 | M4 |
| 2026-02-19 | Codex | 本地 dry-run 验证 `run_phase_d_multimodal_bundle.sh`（全步骤跳过模式）通过，成功生成 `phase_d_summary.json`。 | M4 |
| 2026-02-19 | Codex | 推进 D-3：`training/backbone_experiments.py` 新增 `TFTLite`（`tft` backbone）并纳入统一 WF/PKF 实验协议；默认 backbone 列表扩展为 `ridge,itransformer,patchtst,tft`。 | M4 |
| 2026-02-19 | Codex | 同步配置与测试：`scripts/run_phase_d_multimodal_bundle.sh`、`.env.example`、`README.md` 更新 `tft` 默认项；新增 `normalize_backbones` 相关测试并通过。 | M4 |
| 2026-02-19 | Codex | 推进 D 阶段汇总闭环：新增 `scripts/summarize_phase_d_results.py`，输出统一 JSON/Markdown 对照表（多骨干 + 消融）。 | M4 |
| 2026-02-19 | Codex | `run_phase_d_multimodal_bundle.sh` 接入统一汇总步骤；新增 `backend/tests/test_phase_d_summary_builder.py` 并通过 dry-run 产出 `phase_d_unified_summary.json/.md`。 | M4 |
| 2026-02-19 | Codex | 完成 E-1：`backend/metrics.py` 增加多模态健康指标；`backend/main.py` 新增 `_extract_multimodal_health`、`/api/v2/monitor/multimodal-health`，并在 `model-status` 返回 `multimodal_health/phase_d_summary`。 | M5 |
| 2026-02-19 | Codex | 完成 E-2：`monitoring/model_ops_scheduler.py` 接入多模态 gate（含 snapshot 落地）；`scripts/validate_no_leakage.py` 增加候选 gate 与 multimodal gate 快照检查（支持 `VALIDATE_*` 配置）。 | M5 |
| 2026-02-19 | Codex | 完成 E-2 回归：新增 `backend/tests/test_model_ops_scheduler_multimodal_gate.py`、`backend/tests/test_validate_no_leakage_gate_checks.py`、`backend/tests/test_main_multimodal_health_extract.py` 并通过定向测试。 | M5 |
| 2026-02-19 | Codex | 完成 E-3：新增 `scripts/rehearse_server_release_rollback.sh`，本地 `DRY_RUN=1` 预演通过；支持联动 `run_phase_d_multimodal_bundle.sh` 的演练模式。 | M5 |
| 2026-02-19 | Codex | 完成 E-4：`README.md` 与 `REPOSITORY_ARCHITECTURE_GUIDE_ZH.md` 同步新增多模态监控 endpoint、门禁配置与发布回滚预演流程。 | M5 |
| 2026-02-19 | Codex | 补充本地预演兼容：`scripts/server_preflight.sh` 新增 `REQUIRE_SCREEN` 开关（默认仍为 1）；验证 `REQUIRE_SCREEN=0` 预检通过，完整 `DRY_RUN=0` 演练在本机因缺 `screen` 停于 `server_down`。 | M5 |
| 2026-02-19 | User+Codex | 完成本机依赖安装：`screen/postgresql/redis-server`，并同步本地 `monitor` DB 角色口令与 `.env`。 | M5 |
| 2026-02-19 | Codex | 完成服务脚本解释器统一：`server_up/server_preflight/server_readiness` 支持 `PYTHON_BIN`，默认优先 `.venv/bin/python`，避免系统 python 缺包导致启动失败。 | M5 |
| 2026-02-19 | Codex | 完成真实预演闭环：`DRY_RUN=0` 通过 `server_preflight -> server_up -> server_readiness -> phase_d_bundle(dry) -> server_down` 全链路。 | M5 |
| 2026-02-19 | Codex | 新增 `scripts/ensure_postgres_access.sh`，将 `.env DATABASE_URL` 的角色/库同步动作脚本化（通过 `sudo -u postgres` 执行）。 | M5 |
| 2026-02-19 | Codex | 校验 `ensure_postgres_access.sh`：修复临时脚本权限后已可成功同步 `monitor/monitor` 角色与数据库。 | M5 |
| 2026-02-19 | User+Codex | 新增需求：按训练目标盘点“数据窗口 + 数据种类”完整性，缺失则补采集代码并持续维护到计划中。 | M5/F |
| 2026-02-19 | Codex | 完成 F-1 首轮审计：全量窗口与训练审计显示 `market_bars/orderbook_l2/funding_rates/onchain/social` 覆盖率均接近 0，当前为冷启动空库状态。 | F |
| 2026-02-19 | User+Codex | 需求约束补充：优先复用仓库内已有采集代码；仅在缺口处新增脚本，并校验旧逻辑是否可直接复用。 | F |
| 2026-02-19 | Codex | 推进 F-2/F-3/F-4：新增 `scripts/ingest_binance_aux_signals.py`、`scripts/backfill_orderbook_from_market_bars.py`、`scripts/remediate_liquid_data_gaps.sh`；编排层复用既有 `ingest_bitget_market_bars/backfill_social_history/import_social_events_jsonl/audit_*`。 | F |
| 2026-02-19 | Codex | 新增配套单测：`backend/tests/test_ingest_binance_aux_signals.py`、`backend/tests/test_backfill_orderbook_from_market_bars.py`，用于保证关键转换逻辑可复现。 | F |
| 2026-02-19 | Codex | 完成本地联调预演：`ingest_binance_aux_signals` dry-run/实跑、`backfill_orderbook_from_market_bars` dry-run/实跑、`remediate_liquid_data_gaps.sh` 分阶段 smoke-run 全部通过。 | F |
| 2026-02-19 | Codex | 为避免污染本地库，已清理联调产生的测试行（`market_bars/orderbook_l2/funding_rates/onchain_signals` 回到 0）。 | F |
| 2026-02-19 | User+Codex | 部署策略调整：服务器位于国内且网络受限，默认改为“本地采集后上传/通过 GitHub 分发”。 | F |
| 2026-02-19 | Codex | 推进 F-7：新增 `scripts/export_liquid_data_csv.py`、`scripts/import_liquid_data_csv.py`、`scripts/build_offline_data_bundle.sh`、`scripts/import_offline_data_bundle.sh`、`scripts/server_import_offline_bundle.sh`；并让 `orchestrate_event_social_backfill.py` 支持 `PYTHON_BIN`。 | F |
| 2026-02-19 | Codex | F-7 本地验证通过：`build_offline_data_bundle.sh`（空数据 smoke）可产出 `offline_data_bundle_*.tar.gz`；`import_offline_data_bundle.sh`（非严格 asof）可完成导入与审计。 | F |
| 2026-02-19 | Codex | 推进 F-8：新增 `scripts/ingest_binance_derivatives_signals.py`，接入 Binance 衍生品长短仓比/主动买卖比/基差采集并统一落 `onchain_signals`。 | F |
| 2026-02-19 | Codex | 推进 F-9：新增 `scripts/audit_required_data_readiness.py` 与 `scripts/collect_required_data.sh`，打通“采集->审计->离线包”一键闭环。 | F |
| 2026-02-19 | Codex | 同步离线链路升级：`build_offline_data_bundle.sh` 增加 derivatives 采集与 required-readiness 审计产物；`.env.example`、`README.md` 同步更新。 | F |
| 2026-02-19 | Codex | 推进 G-1：新增 `scripts/gate_training_profitability.py`，支持按最近回测结果做收益门禁并输出快照。 | G |
| 2026-02-19 | Codex | 推进 G-2：新增 `scripts/launch_dual_a100_training.sh`，并增强 `scripts/train_gpu_stage2.py` 的 GPU/NVLINK 预检与结果回传。 | G |
| 2026-02-19 | Codex | 推进 G-3：`scripts/continuous_ops_loop.py` 新增“首训成功后再跑模拟盘 + 收益门禁失败触发微调”机制；补充 `.env.example/README.md` 说明。 | G |
| 2026-02-20 | Codex | 本地实跑 `collect_required_data` 主链路：完成 2018-now 的 5m/1h 市场、aux、orderbook 采集；识别社媒与事件编排在默认参数下的长耗时/限流问题。 | F |
| 2026-02-20 | Codex | 修复数据链路阻塞点：`orchestrate_event_social_backfill.py` 增加事件源开关（disable-gdelt/google/rss/source-balance）与 `event-day-step`；`collect_required_data.sh` 同步参数化。 | F |
| 2026-02-20 | Codex | 修复社媒导入问题：`backend/v2_repository.py` 修复 `ingest_events` 的 `fetchone` 时序 bug（savepoint release 后取结果导致异常）；导入链路恢复。 | F |
| 2026-02-20 | Codex | 完成 recent-window 事件/社媒补采与导入：事件 2589、社媒 480；`audit_required_data_readiness` 的社媒/事件门禁转绿。 | F |
| 2026-02-20 | Codex | 升级衍生品采集：`ingest_binance_derivatives_signals.py` 新增低频指标 5m 前向扩展 + `annualized_basis_rate` 派生补齐；derivatives 覆盖门禁转绿，`ready=true`。 | F |
| 2026-02-20 | Codex | 新增 `TRAINING_PIPELINES_SUMMARY_ZH.md`，系统梳理全部训练链路（数据使用、特征提取、标签定义、产物与限制），用于后续服务器训练与排障对照。 | G |
| 2026-02-20 | Codex | 更新 `TRAINING_PIPELINES_SUMMARY_ZH.md` 的“当前已有数据”章节，补充最新审计口径下的数据种类、覆盖率与 420d/2018-now 双窗口状态。 | G |
| 2026-02-20 | Codex | 完成 multi-horizon Phase 0-5 代码重构：训练/推理/信号/仓位/执行/监控/回滚全链路落地，新增 allocator、execution policy、registry rollback 与监控端点。 | MH |
| 2026-02-20 | Codex | 完成全量回归：`pytest -q` -> `199 passed, 2 skipped`；同步更新 `README.md`、`TRACKING.md`、`PRODUCTION_READINESS_CHECKLIST.md` 与本计划第 12 章交付清单。 | MH |
| 2026-02-20 | Codex | 继续阶段（MH-6）：新增 `multi_model_meta` 训练模式（多子模型+meta），并补齐 `symbol+horizon` registry 的 `candidate -> promote -> rollback` 生命周期接口与测试。 | MH |
| 2026-02-21 | Codex | 完成 MH-7：新增 `scripts/run_multi_horizon_upgrade_bundle.py` 并实跑严格模式（`--require-db-gates --strict`），统一产出升级核验 artifact。 | MH |
| 2026-02-21 | Codex | 修复 `scripts/merge_feature_views.py` 在 `market_latent/social_text_latent` 缺表时崩溃的问题，改为缺表降级继续构建 `feature_matrix_main`。 | MH |

## 8. 明日优先级（滚动 3 项）

1. 进入 Phase G 实跑：按收益门禁脚本触发首轮训练（双卡服务器执行），产出第一版 `backtest_runs` 与 gate 快照。
2. 接通“首训通过后模拟盘”闭环：完成 `continuous_ops_loop` + `paper_trading_daemon` 的端到端联调与回放验证。
3. 执行仓库清理与结构收敛：删除无用工件/临时文件，补齐部署脚本说明后推送 GitHub。
2. 复核 `artifacts/data_collection/run_*/readiness_after.json` 的 `missing_data_kinds`，对残余缺口按 `recommended_commands` 定点补采。
3. 产出并上传 `offline_data_bundle_*.tar.gz`，在服务器导入后执行 Phase-D 训练与评估闭环。

## 9. 本文件使用方式（提醒自己）

- 每次执行任务前先看第 4 节，定位 `IN_PROGRESS` 和 `TODO`。
- 每次提交后更新第 4 节状态与第 7 节日志，避免“做了但没记录”。
- 如果任务被阻塞，必须在第 6 节登记风险并给出下一步解法。

## 10. 2026+ 数据扩展清单（持续维护）

| 数据类别 | 当前状态 | 训练价值 | 采集路径/代码 | 下一步 |
|---|---|---|---|---|
| 5m 价量（Top10） | 缺口大（空库起步） | 主干特征与标签 | 复用 `scripts/ingest_bitget_market_bars.py` | 先补全 2018-now，再跑 WF 训练 |
| 订单簿 L2（5m代理） | 缺口大 | 微观结构、冲击成本估计 | 新增 `scripts/backfill_orderbook_from_market_bars.py` | 先冷启动代理，再接实时深度 |
| 资金费率（5m对齐） | 缺口大 | 多空拥挤/杠杆状态 | 新增 `scripts/ingest_binance_aux_signals.py` | 与 OI 联动做 regime 特征 |
| OI / OI delta | 真实历史窗口受限 | 杠杆扩张/去杠杆识别 | 新增 `scripts/ingest_binance_aux_signals.py`（含 fallback） | 评估是否接入付费长历史数据源 |
| 社媒帖子+评论 | 缺口大 | 事件驱动/情绪冲击 | 复用 `scripts/backfill_social_history.py` + `scripts/import_social_events_jsonl.py` | 分平台做质量分层与去噪 |
| 多源事件（宏观/监管/交易所） | 已有代码，待全窗口执行 | 事件密度、风险状态机 | 复用 `scripts/build_multisource_events_2025.py` + 导入脚本 | 扩展到 2018-now 分块回填 |
| 衍生品清算/基差/期限结构 | 已接入第一版（Binance futures/data） | 高波动阶段风险控制 | `scripts/ingest_binance_derivatives_signals.py` | 下一步补“清算流/期限结构”与历史长窗口数据源 |
| 稳定币与交易所净流（链上） | 未接入 | 流动性与风险偏好状态 | 计划新增链上采集脚本并落 `onchain_signals` | 先确定免费/付费源与覆盖窗口 |











## 11. 增补执行清单（P0->P2）

### 11.1 目标
- 把“数据采集能力”与“训练链路真正吃到的数据”对齐；让评估/门禁链路稳定产出；把主训练从固定样本上限改成可配置；补齐 backbone 实验链的 tft 分支。

### 11.2 状态追踪（新增）
| ID | 优先级 | 任务 | 当前状态 |
|---|---|---|---|
| P0.1 | P0 | 修复多模态 OOS 评估链路崩溃 | DONE |
| P0.2 | P0 | Liquid 主训练样本上限参数化（替代固定 limit=4000） | DONE |
| P0.3 | P0 | 接入 derivatives 指标进入 Liquid 训练特征 | DONE |
| P0.4 | P0 | 修复 backbone_experiments 的 tft 训练分支 | DONE |
| P1.1 | P1 | 2018-now 全窗口补齐策略（代码+流程） | DONE |
| P1.2 | P1 | 社媒原始明细表落地（可选但建议） | DONE |
| P1.3 | P1 | onchain_signals 扩展与一致化 | DONE |
| P2.1 | P2 | 可选新数据源评估（先做 feasibility report） | DONE |

### 11.3 P0（必须先做：稳定性 & 训练-数据脱节）

#### P0.1 修复多模态 OOS 评估链路崩溃
- 现状：training/eval_multimodal_oos.py 末尾 print 使用未定义变量 wf_basic，导致评估链不稳定。
- 相关：链路 D 的产物为 artifacts/models/multimodal_eval.json
- 要求（验收）：
  1) 脚本能稳定跑完并写出 multimodal_eval.json
  2) 仍保持 WF + Purged KFold 协议不变
- 建议步骤：
  - 定位 wf_basic 来源（应来自 validation.py 的某个 wf config 或返回值）
  - 修正输出字段名/变量名，补一个最小单测或 smoke test（可用短窗口）

#### P0.2 Liquid 主训练样本上限参数化（替代固定 limit=4000）
- 现状：每 symbol 默认 limit=4000，导致“不是全历史全量训练”。
- 要求（验收）：
  1) 支持 CLI 参数（例如 --start --end --max-samples/--limit 或 env）控制取数
  2) 默认行为保持兼容（仍可用 4000 作为默认）
  3) 在训练 manifest 里记录最终采样策略（窗口、limit/max_samples、下采样方式）
- 影响范围：
  - training/main.py
  - training/liquid_model_trainer.py
  - training/feature_pipeline.py::load_liquid_training_batch

#### P0.3 接入 derivatives 指标进入 Liquid 训练特征
- 现状：衍生品增强指标（long/short ratio、taker ratio、basis 等）审计覆盖很高，但“尚未进入训练特征集合”。
- 要求（验收）：
  1) FeaturePipeline.load_liquid_training_batch 能读取并对齐这些指标（5m as-of / latest_before）
  2) LIQUID_FEATURE_KEYS / 合同特征键同步更新（inference/liquid_feature_contract.py）
  3) 保证 no-lookahead（只能使用 timestamp<=t 或 available_at<=t 的可用数据）
  4) 缺失要有 missing_flag，并进入模型输入
- 附：优先接入已列出的 6 个指标，后续再扩展。

#### P0.4 修复 backbone_experiments 的 tft 训练分支
- 现状：_fold_metrics 分支未把 tft 送入 torch 训练分支，tft 被标记 unsupported。
- 要求（验收）：
  1) tft 能按其它神经骨干一样跑完训练+评估并写入 backbone_suite_latest.json
  2) 明确 torch 依赖缺失时的降级行为（提示 + 跳过）
- 影响范围：
  - training/backbone_experiments.py

### 11.4 P1（数据层补齐：从“短中期就绪”到“可全历史/可扩展”）

#### P1.1 2018-now 全窗口补齐策略（代码+流程）
- 现状：full_history_latest 显示 2018-now 覆盖不足，history_window_complete=false。
- 目标：
  - 明确两种模式：
    A) “生产训练窗口模式”（比如 420 天 lookback）继续作为默认
    B) “研究/全历史模式”（2018-now）可选开启，并且 audit 通过后才允许运行
- 要求（验收）：
  1) scripts/collect_required_data.sh 支持按模式回填
  2) scripts/audit_required_data_readiness.py 输出明确的 mode + coverage + 阈值
  3) training/main.py 在全历史模式下自动开启更严格的缺失处理/下采样策略（避免样本选择偏差）

#### P1.2 社媒原始明细表落地（可选但建议）
- 现状：social_posts_raw / social_comments_raw 不存在；训练社媒信号主要来自 events.payload 和 social_text_latent 聚合表。
- 目标：
  - 把“可追溯、可重算”的 raw 社媒落下来（哪怕先只覆盖近 90/180 天）
  - build_text_latent_embeddings.py 支持从 raw 表重算（而不是强依赖 events.payload）
- 要求（验收）：
  1) 定义表 schema（必要字段：platform、id、ts、author、text、engagement、symbol/entity link）
  2) 提供最小可用采集脚本/任务（或对接现有采集器）
  3) 对齐到 5m 桶聚合，并能生成 social_text_latent（含 agg_features）

#### P1.3 onchain_signals 扩展与一致化
- 现状：Liquid 链路目前仅使用 netflow/exchange_netflow/net_inflow 等少量指标；同时衍生品指标也需要一致的 as-of 对齐。
- 目标：
  - 把“指标字典”集中管理（onchain + derivatives），一处定义，多处复用（特征层 & 训练层）
  - audit 与训练读数保持同一套 key list

### 11.5 P2（增量收益项：可选数据/特征）

#### P2.1 可选新数据源评估（先做 feasibility report）
- 候选：
  - open interest / liquidation（强相关于 5m/1h 短周期）
  - options implied vol / skew（偏中周期）
  - 跨交易所资金费率分布、CVD/成交主动性
  - 宏观风险因子（DXY、US10Y、VIX 等，需对齐到 1h/5m）
- 产出：
  - 1 页 feasibility + 采集成本 + 预期增益 + 风险（数据质量/延迟/泄露）

## 12. Multi-Horizon 生产级升级重构总计划（2026-02-20 起，当前主计划）

> 本节为当前执行主计划，覆盖“训练-推理-信号-仓位-执行-监控”闭环；与旧版任务有冲突时，以本节为准。

### 12.1 背景与约束输入（必须同时遵守）

- `TRAINING_PIPELINES_SUMMARY_ZH.md`
  - 重点约束：Liquid 主链历史上存在 `limit=4000` 快速采样习惯；标签为 `fwd_ret_1h/4h - est_cost`；derivatives 已采集但主训练接入要持续校验。
- `PRODUCTION_READINESS_CHECKLIST.md`
  - 必须保留串行硬门禁与 API Gate，并在本次升级中扩展。
- `MULTIMODAL_UPGRADE_PLAN_ZH.md`
  - 在既有多模态路线（对齐/门控/实验统一）上，新增 multi-horizon 与交易层闭环，不替换人工实盘控制原则。

### 12.2 Definition of Done（总目标）

#### A. Multi-horizon alpha
- 统一输出至少 4 个 horizon：`1h / 4h / 1d / 7d`。
- 每个 horizon 必须输出：
  - `expected_return[h]`
  - `signal_confidence[h]`
  - `vol_forecast[h]`（短期可用波动代理，但必须可替换为训练 head）。
- 训练/推理/信号层同一 horizon 的定义、缩放、成本处理必须一致。
- 删除“拍脑袋 horizon 缩放常数”，改为训练/校准产物驱动。

#### B. 组合与执行分层
- 新增 `portfolio allocator`：将各 horizon 视为策略桶，执行风险预算、相关性去重、单桶限额、单币限额，输出统一 `target_position`。
- 新增 `execution policy`：短周期与长周期执行策略分层，并在 paper trading 复用同一路径。

#### C. 门禁 + 可观测 + 可回滚
- 所有关键脚本必须可重跑，并产出可追溯 artifacts（config、窗口、commit hash、指标摘要）。
- 增加监控：
  - 各 horizon 预测分布漂移
  - confidence 校准
  - text/gate 开度（如启用）
  - 成本后 paper PnL 分桶（horizon/币种/桶）
- 保持并扩展“无自动实盘晋级”：实盘 enable/disable 仅人工触发。

### 12.3 分阶段任务与状态（当前执行表）

| Phase | 目标 | 状态 |
|---|---|---|
| Phase 0 | 基线冻结 + 修 bug + CPU smoke | DONE |
| Phase 1 | 数据与特征契约一致性 | DONE |
| Phase 2 | Multi-horizon 标签/训练/校准 | DONE |
| Phase 3 | 信号层与仓位层重构 | DONE |
| Phase 4 | 执行分层 + paper 闭环 | DONE |
| Phase 5 | 监控、回滚、运维文档 | DONE |

### 12.4 Phase 0：基线冻结 + 修 bug（本轮先执行）

#### 目标条目
- `0.1` 修复 `training/eval_multimodal_oos.py` 末尾未定义变量输出问题，保证评估链稳定。
- `0.2` 修复 `training/backbone_experiments.py` 的 `tft` 分支未进入 torch 训练导致 unsupported 的问题。
- `0.3` 新增/完善 smoke tests：无 GPU 环境可跑最小训练+推理+信号生成。

#### 本轮执行记录（2026-02-20）
- `0.1` 状态：`DONE`
  - 现代码已不存在 `wf_basic` 未定义引用；通过 OOS smoke 用例回归验证。
- `0.2` 状态：`DONE`
  - `tft` 已进入 torch 分支；缺 torch 时降级为 `torch_missing`（非 `unsupported_backbone`）。
- `0.3` 状态：`DONE`
  - 新增 CPU smoke 测试：`backend/tests/test_phase0_cpu_smoke.py`
  - 覆盖链路：最小训练（`train_multimodal`）-> 模型推理（`LiquidModelService`）-> 信号生成（`/signals/generate` 逻辑）。

#### 基线冻结（Phase 0 artifact）
- 产物：`baseline_snapshots/liquid_baseline_freeze_20260220T151336Z.json`
- latest：`baseline_snapshots/liquid_baseline_latest.json`
- 摘要：
  - `artifacts=4`
  - `missing_component_refs=3`（需在后续服务器训练补齐 `.pt` 组件）
  - `snapshot_sha256=eaa7eb4d2f626575e1198816a9e41795a73f7929184e54ca27346b2166b89242`

### 12.5 Phase 1：数据与特征契约（训练/推理一致性）

- `1.1` Liquid 训练从固定 `limit=4000` 升级为“窗口优先 + 可配置下采样”。
  - 新增参数：`--start --end` 或 `LIQUID_TRAIN_LOOKBACK_DAYS`。
  - 规则：production mode 默认不用硬 `limit=4000`；fast mode 才允许。
- `1.2` `FeaturePipeline.load_liquid_training_batch` 接入 derivatives 指标并保证 as-of 对齐。
  - 使用 `DERIVATIVE_METRIC_KEY_MAP / DERIVATIVE_FEATURE_KEYS / missing flags`。
  - 扩展 `validate_asof_alignment / validate_no_leakage` 到 derivatives。
- `1.3` 特征契约校验。
  - 增加单测校验：`inference/liquid_feature_contract.py` 与 `training/feature_pipeline.py` 的 key 顺序/维度一致。
  - 不允许线上/线下双处手写漂移，必要时引入 contract-first 生成。

### 12.6 Phase 2：Multi-horizon 标签、训练与校准

- `2.1` 标签扩展到 `1h/4h/1d/7d`，统一 `y_h = fwd_ret_h - cost_h`。
  - `cost_h` 配置化并写入 manifest（fee+slippage+impact + horizon/turnover 系数）。
- `2.2` 模型结构：
  - `A` 单模型多头（共享 backbone + 4 horizon 回归头 + confidence/vol head）；
  - `B` 多模型（每 horizon 子模型 + meta-aggregator）。
  - 至少落地一种 production；两种接口都保留。
- `2.3` confidence calibration。
  - 输出分桶校准摘要，保证阈值可解释性。

### 12.7 Phase 3：信号层与仓位层重构

- `3.1` `/predict/liquid` 升级为多 horizon 返回结构并保持兼容字段。
- `3.2` 信号逻辑改为 cost-aware + risk-normalized：
  - `edge_h = expected_return_h - cost_h`
  - `score_h = edge_h / max(vol_forecast_h, eps)`
  - action 由 `(score_h, confidence_h)` + horizon 阈值决定。
- `3.3` 新增 `portfolio allocator`：
  - 输入 `{score_h, confidence_h, vol_h}` + risk/bucket 配置；
  - 输出 `target_position` + horizon 贡献归因；
  - 约束：总风险、单币、单桶、相关性去重。
- `3.4` 仓位强度映射可插拔：旧逻辑与新 allocator 可 feature flag 切换。

### 12.8 Phase 4：执行分层 + paper trading 闭环

- `4.1` 新增执行器接口：
  - `execute(target_position, current_position, horizon_bucket, liquidity_metrics, orderbook)`
  - 策略：`short_horizon_exec`（更快）与 `long_horizon_exec`（更被动）。
- `4.2` paper trading 必须复用同一执行路径。
  - 记录理论价/成交价/滑点/费用/冲击。
- `4.3` `scripts/gate_training_profitability.py` 升级为 multi-horizon/bucket 统计。
  - Gate 要求：至少 2 个 walk-forward 周期成本后优于 Phase 0 baseline。

### 12.9 Phase 5：监控、回滚、可运维

- `5.1` 扩展 `/api/v2/monitor/*`：horizon performance、prediction drift、confidence calibration、text/gate 开度。
- `5.2` registry 与回滚：按 symbol+horizon 记录 active/candidate，支持一键 rollback。
- `5.3` 文档补齐：README/TRACKING/运维手册，覆盖配置、上线、回滚、故障处理。

### 12.10 硬性工程要求（全阶段）

- 严禁未来泄露：所有 join 必须 as-of / latest_before，并通过 `validate_asof_alignment + validate_no_leakage`。
- 训练-推理特征一致：同一 contract 生成/校验。
- 新配置必须有默认值并写入 manifest。
- 交易影响改动必须具备测试覆盖：
  - predict vs generate_signal 一致性
  - contract 一致性
  - allocator 约束
  - paper trading 事件链路
- 严禁自动实盘晋级路径，实盘仅人工 enable/disable。

### 12.11 阶段交付模板（每个 Phase 必交）

- 代码改动（含 tests）
- artifacts（基线冻结、训练评估、门禁摘要 JSON）
- 文档（命令、环境变量、回滚、监控解释）
- 生产推荐配置（窗口、阈值、risk budget、执行参数）与基于 OOS + paper 数据的理由

### 12.12 Phase 0-5 执行结果（2026-02-20，代码与测试已闭环）

#### Phase 0（DONE）
- changed files：
  - `training/eval_multimodal_oos.py`
  - `training/backbone_experiments.py`（已在此前提交修复 `tft` torch 分支）
  - `backend/tests/test_phase0_cpu_smoke.py`
  - `baseline_snapshots/liquid_baseline_freeze_20260220T151336Z.json`
  - `baseline_snapshots/liquid_baseline_latest.json`
- 如何运行：
  - `pytest -q backend/tests/test_phase0_cpu_smoke.py`
  - `.venv/bin/python scripts/freeze_liquid_baseline.py --label phase0_multi_horizon_freeze`
- tests/gates：
  - `backend/tests/test_phase0_cpu_smoke.py` 通过
  - 基线快照已冻结并更新 latest
- artifacts 与指标摘要：
  - `baseline_snapshots/liquid_baseline_freeze_20260220T151336Z.json`
  - `artifact_count=4`，`missing_component_refs=3`，`snapshot_sha256=eaa7eb4d2f626575e1198816a9e41795a73f7929184e54ca27346b2166b89242`

#### Phase 1（DONE）
- changed files：
  - `training/main.py`
  - `training/liquid_model_trainer.py`
  - `training/feature_pipeline.py`
  - `backend/tests/test_feature_contract_alignment.py`
- 如何运行：
  - `python3 training/main.py --run-once --enable-liquid 1 --liquid-train-mode production --liquid-lookback-days 365`
  - `python3 training/main.py --run-once --enable-liquid 1 --liquid-train-mode fast --liquid-limit 4000 --liquid-max-samples 4000`
  - `pytest -q backend/tests/test_feature_contract_alignment.py backend/tests/test_feature_pipeline_derivatives.py`
- tests/gates：
  - 训练模式切换已覆盖（production 窗口优先、fast 才使用 limit）
  - 合约对齐与 derivatives 对齐用例通过
- artifacts 与指标摘要：
  - `artifacts/audit/asof_alignment_multi_horizon_latest.json`
  - 本机当前状态：`future_leakage_count=0`，但 `feature_snapshots_main_present=false`、`feature_matrix_main_present=false`（数据前置条件未满足）

#### Phase 2（DONE）
- changed files：
  - `training/feature_pipeline.py`
  - `training/liquid_model_trainer.py`
  - `inference/model_router.py`
  - `backend/liquid_model_service.py`
  - `training/eval_multimodal_oos.py`
- 如何运行：
  - `python3 training/main.py --run-once --enable-liquid 1 --liquid-train-mode production --liquid-lookback-days 365`
  - `python3 training/eval_multimodal_oos.py --out artifacts/models/multimodal_eval.json`
- tests/gates：
  - multi-horizon 输出与兼容字段测试通过
  - 评估脚本校准摘要输出测试通过
- artifacts 与指标摘要：
  - 训练产物 manifest 已记录 `horizons`、`cost_config`、`horizon_heads`
  - OOS 结果在本机无完整训练矩阵时不具代表性，需在服务器有数据环境复跑

#### Phase 3（DONE）
- changed files：
  - `backend/schemas_v2.py`
  - `backend/v2_router.py`
  - `backend/portfolio_allocator.py`
  - `backend/tests/test_multi_horizon_signal_logic.py`
  - `backend/tests/test_portfolio_allocator_constraints.py`
  - `backend/tests/test_signal_predict_consistency.py`
- 如何运行：
  - `pytest -q backend/tests/test_multi_horizon_signal_logic.py backend/tests/test_portfolio_allocator_constraints.py backend/tests/test_signal_predict_consistency.py`
  - `PORTFOLIO_ALLOCATOR_MODE=allocator_v2` 调用 `POST /api/v2/portfolio/rebalance`
- tests/gates：
  - cost-aware + risk-normalized 信号逻辑通过
  - allocator 风险预算、单币上限、桶约束测试通过
- artifacts 与指标摘要：
  - 推理输出已包含 `score_horizons`、`edge_horizons`、`action_horizons`
  - `allocator_details` 已进入 rebalance 返回与订单 metadata

#### Phase 4（DONE）
- changed files：
  - `backend/execution_policy.py`
  - `backend/execution_engine.py`
  - `monitoring/paper_trading_daemon.py`
  - `scripts/gate_training_profitability.py`
  - `backend/tests/test_execution_policy_context.py`
  - `backend/tests/test_paper_trading_execution_events.py`
- 如何运行：
  - `pytest -q backend/tests/test_execution_policy_context.py backend/tests/test_paper_trading_execution_events.py`
  - `python3 monitoring/paper_trading_daemon.py --loop --interval-sec 60 --execution-events-file artifacts/paper/paper_execution_events.jsonl`
  - `.venv/bin/python scripts/gate_training_profitability.py --database-url "$DATABASE_URL" --out-json artifacts/ops/training_profitability_gate_multi_horizon_latest.json`
- tests/gates：
  - short/long horizon 执行策略分流测试通过
  - paper execution trace 事件链路测试通过
  - gate 脚本已新增 horizon/bucket/symbol/tail 统计字段
- artifacts 与指标摘要：
  - `artifacts/ops/training_profitability_gate_multi_horizon_latest.json`
  - 当前本机 `runs_total=0`（缺少回测记录），仅验证脚本与字段结构

#### Phase 5（DONE）
- changed files：
  - `backend/main.py`
  - `backend/liquid_model_registry.py`
  - `scripts/rollback_liquid_model.py`
  - `backend/v2_router.py`
  - `README.md`
  - `TRACKING.md`
  - `PRODUCTION_READINESS_CHECKLIST.md`
- 如何运行：
  - `GET /api/v2/monitor/horizon-performance`
  - `GET /api/v2/monitor/prediction-drift`
  - `GET /api/v2/monitor/confidence-calibration`
  - `GET /api/v2/monitor/paper-pnl-buckets`
  - `GET /api/v2/models/liquid/registry`
  - `POST /api/v2/models/liquid/registry/activate`
  - `POST /api/v2/models/liquid/registry/rollback`
  - `python3 scripts/rollback_liquid_model.py --symbol BTC --horizon 1h --operator ops`
- tests/gates：
  - `backend/tests/test_liquid_model_registry.py` 通过
  - 监控与路由变更已在 `pytest -q` 全量验证
- artifacts 与指标摘要：
  - `artifacts/audit/no_leakage_multi_horizon_latest.json`
  - 当前本机 `checked_runs=0`，因此 `leakage_passed=false`；属于数据前置条件缺失，不是代码异常

### 12.13 全量回归结果（本轮）

- `pytest -q`：`203 passed, 2 skipped`
- 关键新增测试（全部通过）：
  - `backend/tests/test_phase0_cpu_smoke.py`
  - `backend/tests/test_feature_contract_alignment.py`
  - `backend/tests/test_multi_horizon_signal_logic.py`
  - `backend/tests/test_portfolio_allocator_constraints.py`
  - `backend/tests/test_execution_policy_context.py`
  - `backend/tests/test_paper_trading_execution_events.py`
  - `backend/tests/test_liquid_model_registry.py`
  - `backend/tests/test_liquid_registry_routes.py`

### 12.16 继续阶段（MH-6：多子模型 + candidate 生命周期）

#### 目标
- 在既有 A 方案（single_model_multihead）基础上，补齐 B 方案显式支持：`multi_model_meta`（每 horizon 子模型 + 线性 meta 聚合器）。
- 补齐 registry 的 `candidate` 语义与手工 `promote-candidate` 流程，保持“无自动实盘晋级”。
- 提升门禁 artifact 可追溯性：收益门禁快照附带 `git` 与 `config`。

#### changed files
- `training/liquid_model_trainer.py`
- `inference/model_router.py`
- `backend/liquid_model_registry.py`
- `backend/v2_router.py`
- `scripts/gate_training_profitability.py`
- `backend/tests/test_model_router_residual_fusion.py`
- `backend/tests/test_liquid_model_registry.py`
- `backend/tests/test_liquid_registry_routes.py`
- `README.md`
- `TRACKING.md`
- `PRODUCTION_READINESS_CHECKLIST.md`
- `TRAINING_PIPELINES_SUMMARY_ZH.md`

#### 如何运行
- B 方案训练：
  - `LIQUID_MULTI_HORIZON_TRAIN_MODE=multi_model_meta python3 training/main.py --run-once --enable-liquid 1 --liquid-train-mode production`
- registry 候选链路：
  - `POST /api/v2/models/liquid/registry/candidate`
  - `POST /api/v2/models/liquid/registry/promote-candidate`
  - `POST /api/v2/models/liquid/registry/rollback`
- 收益门禁（含 git/config）：
  - `.venv/bin/python scripts/gate_training_profitability.py --database-url "$DATABASE_URL" --out-json artifacts/ops/training_profitability_gate_multi_horizon_latest.json`

#### tests/gates
- `pytest -q backend/tests/test_model_router_residual_fusion.py backend/tests/test_liquid_model_registry.py backend/tests/test_liquid_registry_routes.py` 通过。
- `pytest -q` 全量回归通过（见 12.13）。

#### artifacts 与指标摘要
- `artifacts/ops/training_profitability_gate_multi_horizon_latest.json`
  - 当前包含 `summary.horizon_stats/bucket_stats/symbol_stats`
  - 新增 `git.head/head_short/branch/dirty`
  - 新增 `config.window.lookback_hours/limit`

### 12.17 运维自动化补强（MH-7：一键升级核验 bundle）

#### 目标
- 把“关键测试 + 门禁脚本 + git/config 元信息”固化成单命令，可重复执行并写统一 artifact。

#### changed files
- `scripts/run_multi_horizon_upgrade_bundle.py`
- `README.md`
- `PRODUCTION_READINESS_CHECKLIST.md`

#### 如何运行
- `python3 scripts/run_multi_horizon_upgrade_bundle.py --run-full-pytest --database-url "$DATABASE_URL"`
- 上线前严格模式：`python3 scripts/run_multi_horizon_upgrade_bundle.py --run-full-pytest --require-db-gates --strict`

#### tests/gates
- bundle 内置关键测试与全量测试，当前运行结果：
  - `targeted_pytests`: 通过
  - `full_pytest`: 通过（`203 passed, 2 skipped`）
  - `validate_asof_alignment`: 通过（`future_leakage_count=0`，`snapshot_sample_time_match_rate=1.0`）
  - `validate_no_leakage`: 通过（`checked_runs=4`，`violating_runs=0`）
  - `gate_training_profitability`: 通过（默认阈值，`runs_completed=4`，`improved_wf_windows=6`）

#### artifacts 与指标摘要
- `artifacts/upgrade/multi_horizon_upgrade_bundle_latest.json`
  - `status=passed`
  - `summary.steps_total=5`
  - `summary.required_failed=0`
  - `summary.optional_failed=0`
  - 含 `git.head/head_short/branch/dirty` 与每一步命令/耗时/输出尾部
- 本机 gate dry-run 数据准备（用于验证脚本链路，不代表最终生产业绩）：
  - `feature_snapshots_main_rows=43002`
  - `feature_matrix_main_rows=43002`
  - `feature_snapshots_seed_rows=15000`（`lineage_id=train-seed-20260221`）
  - `backtest_seed_rows=4`（`run_name like liquid-seed%`）

### 12.14 推荐生产配置（基于当前可观测结果，保守启动版）

- 说明：本机当前数据库缺少可用 OOS/walk-forward 与持续 paper 记录（`runs_total=0`、`checked_runs=0`），因此此处给出“保守启动配置”；正式阈值需在服务器连续 OOS + paper 产物到齐后二次校准。
- 推荐参数：
  - `LIQUID_TRAIN_MODE=production`
  - `LIQUID_TRAIN_LOOKBACK_DAYS=365`（可在 `180~420` 调优）
  - `LIQUID_MULTI_HORIZON_TRAIN_MODE=single_model_multihead`
  - `PORTFOLIO_ALLOCATOR_MODE=allocator_v2`
  - `ALLOCATOR_SINGLE_SYMBOL_MAX=0.20`
  - `ALLOCATOR_BUCKET_LIMITS=trend=0.55,event=0.70,mean_reversion=0.45`
  - `SIGNAL_SCORE_ENTRY_BY_HORIZON=1h=0.60,4h=0.50,1d=0.35,7d=0.25`
  - `SIGNAL_CONFIDENCE_MIN_BY_HORIZON=1h=0.55,4h=0.55,1d=0.50,7d=0.50`
  - `EXEC_SHORT_MAX_SLIPPAGE_BPS=18`，`EXEC_LONG_MAX_SLIPPAGE_BPS=10`
  - `EXEC_SHORT_MAX_PARTICIPATION=0.35`，`EXEC_LONG_MAX_PARTICIPATION=0.12`
- 理由：
  - 先以低杠杆/低参与率保证执行稳定性与可回滚。
  - 短周期阈值更严格，降低噪声交易与过度换手。
  - 保留人工实盘控制，不存在自动晋级路径。

### 12.15 剩余运维动作（代码已完成，等待数据环境）

1. 在服务器有完整特征矩阵与回测记录后，重跑：
   - `scripts/validate_asof_alignment.py`
   - `scripts/validate_no_leakage.py`
   - `scripts/gate_training_profitability.py`（`min_improved_wf_windows>=2`）
2. 运行至少 14 天 paper trading，收敛分 horizon 的 slippage/cost/turnover，再锁定最终阈值。
3. 保持“无自动实盘晋级”，仅通过人工 `live/enable`/`live/disable` 控制。
