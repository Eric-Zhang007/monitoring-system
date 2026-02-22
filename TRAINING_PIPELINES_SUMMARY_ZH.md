# Strict 重构后 Pipeline 总览（代码事实）

更新时间：2026-02-22

## 1. 四大契约

1. 唯一 schema：`schema/liquid_feature_schema.yaml`
2. 代码生成契约：`schema/codegen_feature_contract.py` -> `features/feature_contract.py`
3. 对齐规则：`features/align.py`（禁止 pad/truncate，schema 不匹配直接报错）
4. 工件规则：`artifacts/manifest.py` + `artifacts/validate.py`

## 2. 特征链路

1. 数值特征构建：`scripts/build_feature_store.py`
   - 输出：`feature_snapshots_main`
   - 每行包含 `feature_values + feature_mask + schema_hash`
   - 默认过滤 synthetic（可显式 `--allow-synthetic`）

2. 文本语义 embedding：`scripts/build_text_embeddings.py`
   - 使用 `sentence-transformers` 本地模型（缺模型直接失败）
   - 输出：`social_text_embeddings`

3. 最终合并：`scripts/merge_feature_views.py`
   - 统一输出：`feature_matrix_main`
   - 字段：`values[L,D 单行 D] + mask + features + schema_hash + synthetic_ratio`

## 3. 训练链路

1. 数据集：`training/datasets/liquid_sequence_dataset.py`
   - 从 `feature_matrix_main` 拉连续序列 `X_values[L,D] + X_mask[L,D]`
   - 缺桶补“全缺失帧”

2. 标签：`training/labels/liquid_labels.py`
   - `1h/4h/1d/7d` 成本后收益

3. 主模型：
   - `models/patchtst.py`
   - `models/multimodal_gate.py`

4. 训练入口：`training/train_liquid.py`
   - 输出 OOS 成本后指标与 gate 分布
   - 打包工件到 `artifacts/models/<model_id>/`

5. VC 训练：`training/train_vc.py`
   - 同样走 manifest 强校验工件链

## 4. 推理链路

1. 序列读取：`inference/feature_reader.py`
2. 推理入口：`inference/main.py`
   - 启动时 validate manifest
   - schema_hash 不一致直接失败
   - 无 fallback

3. 后端服务：
   - Liquid：`backend/liquid_model_service.py`
   - VC：`backend/vc_model_service.py`
   - `/api/v2/predict/vc` 已切换到模型服务，不再使用规则分

## 5. 严格门禁

1. `scripts/audit_required_data_readiness.py`
   - `history_window_complete` 必须为 true
   - `text_bucket_coverage` 低于阈值 fail
   - `feature_missing_ratio` 超阈值 fail
   - `synthetic_ratio > 0`（默认阈值）直接 fail

2. `training/register_candidate_model.py`
   - `--enforce-gates` 默认严格

3. `scripts/validate_no_leakage.py`
   - candidate/multimodal gate 默认要求通过

## 6. 关键测试

1. `tests/test_schema_hash_stable.py`
2. `tests/test_codegen_contract_matches_schema.py`
3. `tests/test_align_row_behavior.py`
4. `tests/test_build_sequence_shape.py`
5. `tests/test_manifest_validation.py`
6. `tests/test_train_infer_parity.py`

当前全量测试状态：`pytest -q` => `211 passed, 2 skipped`（2026-02-22）。
