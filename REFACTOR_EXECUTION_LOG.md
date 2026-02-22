# 一次性生产重构执行记录

## 2026-02-22

### Commit 1（完成）
- [x] 固化强制规则文档：`docs/PRODUCTION_REFACTOR_RULES.md`
- [x] 建立唯一 Schema：`schema/liquid_feature_schema.yaml`
- [x] 建立 schema hash + codegen
- [x] 生成 `features/feature_contract.py`
- [x] 实现 `align_row`（禁止 pad/truncate）
- [x] 实现序列输入契约 `X_values[L,D] + X_mask[L,D]`
- [x] 实现工件 manifest/validate 契约
- [x] 增加门禁测试并跑通（8/8）

### Commit 2~10（完成）
- [x] 删除 61 在线投影和伪序列 reshape 路径（改为全量 schema，并在旧路径抛错）
- [x] 重写 feature_store（去硬置0、去 manual_stat 合成）
- [x] 文本 hash trick -> 语义 embedding（`scripts/build_text_embeddings.py` + `text/embedder.py`）
- [x] 训练主链改为序列模型 + residual gate（`training/train_liquid.py`）
- [x] 推理仅读取 `feature_matrix_main` 序列（`inference/feature_reader.py` + 新 `inference/main.py`）
- [x] VC 删除规则分，接训练模型服务链（`backend/vc_model_service.py`）
- [x] gates 默认严格（`audit_required_data_readiness.py`、`validate_no_leakage.py`、`register_candidate_model.py`）
- [x] 新增 `test_train_infer_parity.py`

### 当前状态与风险
- [x] 新增验收脚本：`scripts/run_strict_e2e_acceptance.sh`
- [x] 语法检查通过（核心新增与改造文件）
- [x] 全量测试通过：`pytest -q` => `211 passed, 2 skipped`
- [x] 旧语义测试已切换为 strict 契约断言（禁止 pad/truncate、禁止 artifact fallback、禁止 phase0 tabular-only 路径）
- [ ] 一键验收脚本在当前环境未跑通：缺少 `psycopg2`（`ModuleNotFoundError: No module named 'psycopg2'`）
- [ ] 运行时依赖 `torch` 与语义 embedding 模型文件：未在当前环境安装/准备，服务会按规则 fail-fast

### 2026-02-22 文本模型配置补充
- [x] 选型固定：`intfloat/multilingual-e5-small`（多语种小模型）
- [x] 新增下载脚本：`scripts/setup_text_encoder.py`
- [x] 默认路径固定：`artifacts/models/text_encoder/multilingual-e5-small`
- [x] `scripts/build_text_embeddings.py` 缺本地目录时给出明确 fail-fast 提示
- [x] `scripts/run_strict_e2e_acceptance.sh` 已改为默认同一路径并进行前置检查
