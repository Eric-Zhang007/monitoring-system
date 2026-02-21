# Production Readiness Checklist（串行硬门禁）

## Phase 0
- [ ] 黑名单关键词扫描返回空
- [ ] `bash -n scripts/server_preflight.sh`
- [ ] `bash -n scripts/server_up.sh`
- [ ] `bash -n scripts/server_readiness.sh`
- [ ] `bash -n scripts/server_down.sh`

## Phase 1
- [ ] 运行 `scripts/audit_full_history_completeness.py`
- [ ] 审计窗口为 `2018-01-01T00:00:00Z` 到当前时间
- [ ] 输出 `summary.history_window_complete=true`
- [ ] 输出 `summary.comment_ratio_ge_10x=true`

## Phase 2
- [ ] 若 Phase 1 任一不达标，运行 `scripts/remediate_full_history.py`
- [ ] 每轮修复后自动重跑 Phase 1 审计
- [ ] 仅当两项摘要字段都为 `true` 才进入 Phase 3

## Phase 3
- [ ] `scripts/validate_asof_alignment.py` 输出 `future_leakage_count=0`
- [ ] 训练快照时间与样本时间一致率 `100%`

## Phase 4
- [ ] `feature_snapshots_main` 生成完成
- [ ] `social_text_latent` 生成完成
- [ ] `market_latent` 生成完成
- [ ] 最终特征矩阵维度 `>=528`
- [ ] `inference/liquid_feature_contract.py` 为统一契约

## Phase 5
- [ ] `training/train_multimodal.py` 完成训练
- [ ] `training/eval_multimodal_oos.py` 完成滚动验证
- [ ] `training/register_candidate_model.py` 注册候选模型

## Phase 6
- [ ] `monitoring/paper_trading_daemon.py` 持续运行
- [ ] 前端可手动启停模拟盘/实盘控制
- [ ] 无自动实盘晋级路径

## Phase 7（Multi-horizon 训练/推理/信号一致性）
- [ ] `1h/4h/1d/7d` 四个 horizon 全部可在 `/api/v2/predict/liquid` 返回
- [ ] 返回字段包含：`expected_return`/`signal_confidence`/`vol_forecast` 三组 horizon map
- [ ] 保留兼容字段：`expected_return_legacy`/`signal_confidence_legacy`/`vol_forecast_legacy`
- [ ] `inference/liquid_feature_contract.py` 与 `training/feature_pipeline.py` 契约一致（顺序/维度）
- [ ] `validate_asof_alignment.py` 输出 `future_leakage_count=0`
- [ ] `validate_no_leakage.py` 输出 `violating_runs=0`

## Phase 8（组合与执行闭环）
- [ ] `portfolio/rebalance` 支持 `allocator_v2`，并可回退 `legacy`
- [ ] allocator 约束生效：总风险预算、单币限额、单桶限额、相关性去重
- [ ] 执行策略分层生效：`short_horizon_exec` / `long_horizon_exec`
- [ ] paper trading 复用同一执行路径并输出 execution trace（理论价/成交价/滑点/费用/冲击）
- [ ] `scripts/gate_training_profitability.py` 输出 horizon/bucket/symbol 维度统计

## Phase 9（监控、回滚、运维）
- [ ] 监控接口新增：horizon performance、prediction drift、confidence calibration、paper pnl buckets
- [ ] model registry 支持 `symbol+horizon` 级 active/candidate 管理
- [ ] 支持人工回滚脚本：`scripts/rollback_liquid_model.py`
- [ ] 生产控制原则不变：仅人工 `live/enable`，不存在自动晋级实盘路径
- [ ] 运行 `scripts/run_multi_horizon_upgrade_bundle.py` 并产出汇总 artifact
- [ ] 上线前执行严格模式：`python3 scripts/run_multi_horizon_upgrade_bundle.py --run-full-pytest --require-db-gates --strict`

## API Gate
- [ ] `GET /api/v2/monitor/history-completeness`
- [ ] `GET /api/v2/monitor/alignment`
- [ ] `GET /api/v2/monitor/social-throughput`
- [ ] `GET /api/v2/monitor/model-status`
- [ ] `GET /api/v2/monitor/paper-performance`
- [ ] `GET /api/v2/monitor/horizon-performance`
- [ ] `GET /api/v2/monitor/prediction-drift`
- [ ] `GET /api/v2/monitor/confidence-calibration`
- [ ] `GET /api/v2/monitor/paper-pnl-buckets`
- [ ] `GET /api/v2/models/liquid/registry`
- [ ] `POST /api/v2/models/liquid/registry/candidate`
- [ ] `POST /api/v2/models/liquid/registry/promote-candidate`
- [ ] `POST /api/v2/models/liquid/registry/activate`
- [ ] `POST /api/v2/models/liquid/registry/rollback`
- [ ] `POST /api/v2/control/live/enable`
- [ ] `POST /api/v2/control/live/disable`
