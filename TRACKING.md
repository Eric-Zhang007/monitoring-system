# Tracking

## 当前文档状态（2026-02-22）

1. 文档已按代码现实合并精简，主入口为：
   - `README.md`
   - `TRAINING_PIPELINES_SUMMARY_ZH.md`
2. 以下文档已并入上述两份并移除：
   - `PRODUCTION_READINESS_CHECKLIST.md`
   - `MULTIMODAL_UPGRADE_PLAN_ZH.md`
   - `SERVER_PREP_PLAN_ZH.md`
   - `REPOSITORY_ARCHITECTURE_GUIDE_ZH.md`
   - `SYSTEM_REMEDIATION_BLUEPRINT.md`
   - `SYSTEM_REMEDIATION_BLUEPRINT_ZH.md`
3. 历史冻结记录保留在：
   - `baseline_snapshots/BASELINE_FREEZE_2026-02-19_ZH.md`

## 当前工程判断

1. 主链路已切到 strict 契约（单一 schema、无 pad/truncate、无 fallback、序列 parity）。
2. 当前测试状态：`pytest -q` => `211 passed, 2 skipped`（2026-02-22）。
3. 运行时仍需准备 `torch` 与文本 embedding 模型文件，缺失时会按规则 fail-fast。
