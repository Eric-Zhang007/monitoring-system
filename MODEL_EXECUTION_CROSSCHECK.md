# 模型与执行逐项核对（代码事实）

更新时间：2026-02-22

## 执行（对应 `执行.txt`）
- [x] COMMIT 1 契约模型 + FSM + 契约测试  
  代码：`backend/execution_models.py`, `backend/execution_fsm.py`, `tests/test_execution_fsm.py`, `tests/test_execution_models_schema.py`
- [x] COMMIT 2 OMS 表结构 + `orders_sim` 结构化列  
  代码：`scripts/init_db.sql`
- [x] COMMIT 3 repository OMS 化（decision/child/fill/recon/parent 聚合）  
  代码：`backend/repository_modules/execution_oms_store.py`, `backend/execution_store.py`
- [x] COMMIT 4 `ExecutionEngine.run_decision` OMS 驱动编排 + planner 接入  
  代码：`backend/execution_engine.py`, `backend/execution_planner.py`
- [x] COMMIT 5 adapter 契约统一（paper/coinbase/bitget）  
  代码：`backend/execution_adapters/base.py`, `backend/execution_engine.py`（具体 adapter 实现）
- [x] COMMIT 6 router 串联 decision/parent/child/fill/trace  
  代码：`backend/v2_router.py`
- [x] COMMIT 7 reconciliation daemon + kill switch 触发链路  
  代码：`monitoring/reconciliation_daemon.py`
- [x] COMMIT 8 positions/pnl accounting 闭环  
  代码：`backend/position_accounting.py`, `backend/execution_engine.py`
- [x] COMMIT 9 执行观测指标与结构化日志  
  代码：`backend/metrics.py`, `backend/execution_engine.py`
- [x] COMMIT 10 adapter 合同测试 + paper 端到端  
  测试：`tests/test_execution_adapter_contract.py`, `tests/test_execution_e2e_paper.py`

补充修复（本轮）：
- [x] fills 幂等键防重放：`execution_fills.fill_key` + unique index  
  代码：`scripts/init_db.sql`, `backend/repository_modules/execution_oms_store.py`
- [x] marketable_limit 残量补单与 TWAP 间隔执行  
  代码：`backend/execution_engine.py`

## 模型（对应 `模型.txt`）
- [x] COMMIT 1 输出契约 + backbone/head 契约与注册  
  代码：`models/outputs.py`, `models/backbones/base.py`, `models/backbones/registry.py`, `models/heads/dist_head.py`
- [x] COMMIT 2 真 PatchTST（RevIN + patchify + mask）  
  代码：`models/backbones/patchtst.py`, 测试：`tests/test_patchtst_patchify.py`
- [x] COMMIT 3 iTransformer 主干接入 strict  
  代码：`models/backbones/itransformer.py`, 测试：`tests/test_itransformer_shapes.py`
- [x] COMMIT 4 TFT 主干接入 strict  
  代码：`models/backbones/tft.py`, 测试：`tests/test_tft_minimal.py`
- [x] COMMIT 5 多模态序列融合（text tower + quality + gate）  
  代码：`models/text_tower.py`, `models/quality_encoder.py`, `models/heads/dist_head.py`
- [x] COMMIT 6 组合损失（分布/分位数/方向/gate）  
  代码：`training/losses/liquid_losses.py`, `training/train_liquid.py`
- [x] COMMIT 7 walk-forward + purged 强制切分  
  代码：`training/splits/walkforward_purged.py`, `training/train_liquid.py`
- [x] COMMIT 8 多维 OOS 指标与校准  
  代码：`training/metrics/liquid_metrics.py`, `training/train_liquid.py`
- [x] COMMIT 9 成本统一（labels/oos/action 同源）  
  代码：`cost/cost_profile.py`, `training/labels/liquid_labels.py`, `training/validation.py`, `backend/v2_router.py`
- [x] COMMIT 10 推理 confidence/vol 来自模型输出 + 校准器  
  代码：`backend/liquid_model_service.py`, `training/calibration/calibrate.py`
- [x] COMMIT 11 NIM 失败即 missing（不伪 embedding）  
  代码：`backend/nim_integration.py`, `scripts/merge_feature_views.py`
- [x] COMMIT 12 断路 legacy/toy 主链路  
  代码：`inference/model_router.py`, `training/liquid_model_trainer.py`, `training/backbone_experiments.py`, `backend/v2_repository.py`
- [x] COMMIT 13 VC 训练/推理特征契约统一  
  代码：`schema/vc_feature_schema.yaml`, `vc/feature_spec.py`, `training/train_vc.py`, `backend/vc_model_service.py`
- [x] COMMIT 14 训练→打包→推理集成门禁  
  测试：`tests/test_strict_train_artifact_can_infer.py`, `tests/test_train_infer_parity.py`

补充修复（本轮）：
- [x] 修复 strict 训练最小窗口参数被硬钳制导致无 fold  
  代码：`training/train_liquid.py`
- [x] 修复推理端 quantile 维度判断错误（2D/3D 兼容）  
  代码：`backend/liquid_model_service.py`
- [x] 修复执行测试 torch stub 污染全局模块问题  
  代码：`tests/test_execution_e2e_paper.py`

## 端到端验证（本轮）
- [x] 模型严格链路：`12 passed`  
  命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_strict_train_artifact_can_infer.py tests/test_model_output_contract.py tests/test_backbone_registry.py tests/test_dist_head_shapes.py tests/test_patchtst_patchify.py tests/test_itransformer_shapes.py tests/test_tft_minimal.py tests/test_gate_behavior.py tests/test_vc_train_infer_parity.py`
- [x] parity：`1 passed`  
  命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_train_infer_parity.py`
- [x] 执行+模型联合链路：`46 passed, 1 warning`  
  命令：见 `EXECUTION_OMS_PROGRESS.md` 最新验证段落。
