# 服务器准备计划（修正版，串行）

## 固定顺序
1. 历史数据完整性
2. 时间对齐
3. 数据修复
4. 特征重建
5. 训练管线
6. 模拟盘
7. 前端总控

## Phase 0
- 仅保留四个运行脚本：
  - `scripts/server_preflight.sh`
  - `scripts/server_up.sh`
  - `scripts/server_readiness.sh`
  - `scripts/server_down.sh`
- 文档以 `README.md` 为唯一入口。

## Phase 1
- 运行 `scripts/audit_full_history_completeness.py`
- 时间窗口固定：`2018-01-01T00:00:00Z` 到当前
- 不允许缩窗替代

## Phase 2
- 运行 `scripts/remediate_full_history.py`
- 修复顺序固定：市场 -> 订单簿/资金费率/链上 -> 帖子 -> 评论
- 评论/帖子比率目标：`>=10`

## Phase 3
- 主时间轴：5min bar
- 全模态采用 `available_at <= bar_ts` 的 as-of 对齐
- 运行 `scripts/validate_asof_alignment.py`，未来泄露必须为 0

## Phase 4
- 运行特征构建脚本，输出 `feature_snapshots_main`
- 契约版本统一为主线版本

## Phase 5
- 训练入口：`training/train_multimodal.py`
- 验证入口：`training/eval_multimodal_oos.py`
- 注册入口：`training/register_candidate_model.py`

## Phase 6
- 守护入口：`monitoring/paper_trading_daemon.py`
- 仅人工控制实盘开关，不自动晋级
