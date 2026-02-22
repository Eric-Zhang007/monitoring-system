# Artifacts

本目录存放运行产物与审计结果，例如：

1. `artifacts/audit/*`：数据完整性与训练就绪度审计。
2. `artifacts/models/*`：候选模型与评估结果。
3. `artifacts/phase63/*`：阶段性 hard metrics / parity 结果。
4. `artifacts/ops/*`：持续运行状态与门禁快照。

说明：

1. 这些文件用于运行追溯与状态验证。
2. 不建议把临时大文件直接纳入 git 版本历史。
