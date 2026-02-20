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

## API Gate
- [ ] `GET /api/v2/monitor/history-completeness`
- [ ] `GET /api/v2/monitor/alignment`
- [ ] `GET /api/v2/monitor/social-throughput`
- [ ] `GET /api/v2/monitor/model-status`
- [ ] `GET /api/v2/monitor/paper-performance`
- [ ] `POST /api/v2/control/live/enable`
- [ ] `POST /api/v2/control/live/disable`
