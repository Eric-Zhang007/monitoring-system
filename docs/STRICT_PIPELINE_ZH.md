# Strict Pipeline 总览（代码事实）

更新时间：2026-02-22

## 1. 四大契约

1. 唯一 schema：`schema/liquid_feature_schema.yaml`
2. codegen 契约：`schema/codegen_feature_contract.py` -> `features/feature_contract.py`
3. 对齐规则：`features/align.py`（禁 pad/truncate，schema mismatch 直接报错）
4. 工件规则：`artifacts/manifest.py` + `artifacts/validate.py`

## 2. 特征链路

1. 数值特征：`scripts/build_feature_store.py` -> `feature_snapshots_main`
2. 文本 embedding：`scripts/build_text_embeddings.py` -> `social_text_embeddings`
3. 合并矩阵：`scripts/merge_feature_views.py` -> `feature_matrix_main`

## 3. 训练链路

1. 数据集：`training/datasets/liquid_sequence_dataset.py`
2. 主训练：`training/train_liquid.py`
   - backbone：`patchtst / itransformer / tft`
   - 输出：strict 工件目录（manifest/weights/schema_snapshot/training_report）
3. VC 训练：`training/train_vc.py`（同样 strict 工件链）

## 4. 推理链路

1. 序列读取：`inference/feature_reader.py`
2. 推理入口：`inference/main.py`
3. 服务加载：
   - Liquid：`backend/liquid_model_service.py`
   - VC：`backend/vc_model_service.py`

## 5. 关键门禁

1. `scripts/audit_required_data_readiness.py`
2. `scripts/validate_no_leakage.py`
3. `tests/test_train_infer_parity.py`
4. `tests/test_strict_train_artifact_can_infer.py`
