# Monitoring System（Strict Refactor）

本仓库已切换为一次性生产级重构模式：

1. 单一特征真相源：`schema/liquid_feature_schema.yaml`
2. 禁止 pad/truncate，缺失通过 `mask` 表达，schema 不匹配直接失败
3. 主模型输入固定为 `X_values[L,D] + X_mask[L,D]`
4. 模型工件必须 `manifest + weights + schema_snapshot + training_report`，启动强校验失败即退出
5. 推理仅从 `feature_matrix_main` 读取序列

## 关键入口

1. 特征契约 codegen
```bash
python3 schema/codegen_feature_contract.py --schema schema/liquid_feature_schema.yaml --out features/feature_contract.py
```

2. 构建特征矩阵
```bash
python3 scripts/build_feature_store.py --start 2018-01-01T00:00:00Z
python3 scripts/setup_text_encoder.py --model-id intfloat/multilingual-e5-small --out-dir artifacts/models/text_encoder/multilingual-e5-small
python3 scripts/build_text_embeddings.py --start 2018-01-01T00:00:00Z --model-path "$TEXT_EMBED_MODEL_PATH"
python3 scripts/merge_feature_views.py --start 2018-01-01T00:00:00Z
```

3. 训练
```bash
python3 training/train_liquid.py --model-id liquid_main --out-dir artifacts/models/liquid_main
python3 training/train_vc.py --model-id vc_main --out-dir artifacts/models/vc_main
```

4. 推理
```bash
python3 inference/main.py --symbol BTC --end-ts 2026-02-22T00:00:00Z
```

5. 一键验收
```bash
bash scripts/run_strict_e2e_acceptance.sh
```

## 文本模型部署建议（服务器）

1. 固定模型：`intfloat/multilingual-e5-small`（多语种、小体量、384 维）。
2. 服务器预部署（仅一次）：
```bash
python3 scripts/setup_text_encoder.py --model-id intfloat/multilingual-e5-small --out-dir artifacts/models/text_encoder/multilingual-e5-small
```
3. 运行前环境变量：
```bash
export TEXT_EMBED_MODEL_PATH=artifacts/models/text_encoder/multilingual-e5-small
```
4. `scripts/build_text_embeddings.py` 仅接受本地目录；目录缺失会直接失败，不会 fallback。

## 契约与执行记录

1. 强制规则：`docs/PRODUCTION_REFACTOR_RULES.md`
2. 执行追踪：`REFACTOR_EXECUTION_LOG.md`
