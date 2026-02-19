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
| F-1 | Phase F | 数据窗口与模态完整性盘点（训练视角） | `scripts/audit_full_history_completeness.py`, `scripts/audit_training_data_completeness.py` | 输出缺口报告 + 分模态优先级 | IN_PROGRESS | 2026-02-19 | 2026-02-20 | - |
| F-2 | Phase F | 衍生品信号采集（funding + OI proxy） | `scripts/ingest_binance_aux_signals.py` | `funding_rates/onchain_signals` 非零覆盖 | IN_PROGRESS | 2026-02-19 | 2026-02-21 | - |
| F-3 | Phase F | 订单簿代理回填（用于冷启动） | `scripts/backfill_orderbook_from_market_bars.py` | `orderbook_l2` 冷启动窗口可用 | IN_PROGRESS | 2026-02-19 | 2026-02-21 | - |
| F-4 | Phase F | 缺口修复编排脚本（可直上服务器） | `scripts/remediate_liquid_data_gaps.sh` | 一键执行并输出审计报告 | IN_PROGRESS | 2026-02-19 | 2026-02-21 | - |
| F-5 | Phase F | 2026+ 额外数据路线评估与实施清单 | `MULTIMODAL_UPGRADE_PLAN_ZH.md`, `README.md` | 给出可执行的数据扩展路线图 | IN_PROGRESS | 2026-02-19 | 2026-02-20 | - |
| F-6 | Phase F | 额外模态采集实现（优先复用现有脚本） | `scripts/orchestrate_event_social_backfill.py`, `scripts/build_multisource_events_2025.py` | 新增数据源不破坏既有回填链路 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |
| F-7 | Phase F | 本地采集 -> 服务器离线导入链路 | `scripts/build_offline_data_bundle.sh`, `scripts/import_offline_data_bundle.sh`, `scripts/server_import_offline_bundle.sh` | 服务器无外网时仍可完成数据导入与训练 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |
| F-8 | Phase F | 衍生品增强数据采集（长短仓比/主动买卖比/基差） | `scripts/ingest_binance_derivatives_signals.py` | `onchain_signals` 新增 derivatives metric 覆盖并可审计 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |
| F-9 | Phase F | 必要数据一键采集与就绪审计总入口 | `scripts/collect_required_data.sh`, `scripts/audit_required_data_readiness.py` | 可一键执行采集->复审->离线包，全过程落产物 | IN_PROGRESS | 2026-02-19 | 2026-02-22 | - |
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

## 8. 明日优先级（滚动 3 项）

1. 本地执行 `bash scripts/collect_required_data.sh`（先分块 `INGEST_MAX_CHUNKS/EVENT_MAX_CHUNKS`，再全窗口）直至 `readiness_after.json` 过门禁。
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
