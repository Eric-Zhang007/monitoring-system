# 一次性生产级重构强制规则

## Rule-1：唯一特征真相源
全仓库只允许一个特征 Schema 文件：`schema/liquid_feature_schema.yaml`。任何 key list 不允许手写分叉。

## Rule-2：禁止 pad/truncate
训练与推理都不允许对特征向量做“少了补 0、多了截断”。缺失必须通过 mask 显式表达；schema 不匹配要直接报错并停止服务/训练。

## Rule-3：真时序输入
主模型输入必须是连续时间桶序列 `X_values[L,D] + X_mask[L,D]`，严禁把扁平向量 reshape 成伪序列。

## Rule-4：工件强一致
工件必须包含 manifest + 权重 + schema snapshot + 训练报告；推理启动必须校验“文件存在 + schema_hash 一致”，否则直接失败退出（不允许静默降级）。

## Rule-5：线上吃全量特征
推理只从数据库的 `feature_matrix_main` 读取完整 D 维序列；不允许线上临时拼 61 维 online features 再喂模型。

## Rule-6：禁合成回填入主训练集
orderbook/onchain/social 的 synthetic/backfill 只能进入 `*_synthetic` 表或带强标记字段；默认训练数据过滤掉 synthetic。

## Rule-7：VC 必须闭环
VC 线上不允许规则分作为最终输出；必须调用训练得到的 VC 模型服务链路（同样受工件契约约束）。
