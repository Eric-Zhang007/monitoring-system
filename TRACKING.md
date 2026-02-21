# TRACKING

## 当前升级追踪主文件
1. 多模态升级与重构请优先维护：`MULTIMODAL_UPGRADE_PLAN_ZH.md`
2. 本文件保留高层里程碑摘要；详细任务状态、风险和变更记录统一在上述文件维护。

## 当前有效里程碑
1. Phase 0：仓库运行入口与文档已统一，禁词扫描纳入门禁。
2. Phase 1：2018-01-01 到当前的全窗口完整性审计脚本落地。
3. Phase 2：全窗口缺口修复编排与评论补抓模式落地。
4. Phase 3：训练与推理 as-of 对齐一致，泄露校验脚本落地。
5. Phase 4：主线特征存储与 latent 视图构建链路落地。
6. Phase 5：多模态训练、评估、候选注册流程落地。
7. Phase 6：持续模拟盘守护与人工实盘控制接口落地。
8. Phase MH-0：基线冻结 + bug 修复 + 无 GPU smoke（DONE，2026-02-20）。
9. Phase MH-1：训练窗口优先采样 + 特征契约一致性（DONE，2026-02-20）。
10. Phase MH-2：`1h/4h/1d/7d` multi-horizon 标签/训练头/校准摘要（DONE，2026-02-20）。
11. Phase MH-3：信号层 cost-aware/risk-normalized + allocator_v2（DONE，2026-02-20）。
12. Phase MH-4：执行分层 + paper execution trace + gate 统计扩展（DONE，2026-02-20）。
13. Phase MH-5：监控扩展 + symbol/horizon registry 回滚 + 文档补齐（DONE，2026-02-20）。
14. Phase MH-6：多子模型+meta 训练模式支持 + candidate/promote 生命周期（DONE，2026-02-20）。
15. Phase MH-7：一键升级核验 bundle 与统一 artifact（DONE，2026-02-21）。
