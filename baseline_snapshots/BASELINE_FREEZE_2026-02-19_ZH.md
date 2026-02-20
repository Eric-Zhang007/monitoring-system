# Liquid 基线冻结记录（2026-02-19）

## 冻结目的

- 固化当前生产基线模型工件、关键训练默认参数与特征契约维度。
- 为后续 Phase A/B/C 的实验对照提供可回放锚点。

## 执行命令

```bash
python3 scripts/freeze_liquid_baseline.py \
  --out baseline_snapshots/liquid_baseline_freeze_20260219T040000Z.json
```

## 产物

- 快照文件：`baseline_snapshots/liquid_baseline_freeze_20260219T040000Z.json`
- 最新别名：`baseline_snapshots/liquid_baseline_latest.json`
- 快照签名：`b067ba3ecf5e478c4b15b9a31b001eddf9c42424d21ecef1c16c87d751ec6819`

## 模型工件哈希

| 文件 | SHA256 |
|---|---|
| `backend/models/liquid_btc_lgbm_baseline_v2.json` | `669641f592c89d978fc6216a71b66d07b1286e12096bc898306a91c99049fdbb` |
| `backend/models/liquid_eth_lgbm_baseline_v2.json` | `a56c1e1b6c5ab717630f55708cb74d3b34824bd27b2c2bcd1c5ccf893c9e0acd` |
| `backend/models/liquid_sol_lgbm_baseline_v2.json` | `49913173b3015bde0916855fe53edd1486b361697750fa8bb3f00652eba2bafe` |
| `backend/models/liquid_ttm_ensemble_v2.1.json` | `2bc1c7735f1473c6cb43eeaa217624cc0b49b5221ac016ae5331168f360eb4ac` |

## 冻结时关键事实

- 特征契约（`FEATURE_PAYLOAD_SCHEMA_VERSION=main`）：
  - `base_v2_dim=49`
  - `manual_dim=400`
  - `latent_dim=128`
  - `full_dim=528`
  - `effective_training_dim=528`
- 脚本在无 `numpy` 环境下可运行（使用静态解析回退）。
- `multimodal_defaults` 已纳入 `MULTIMODAL_FUSION_MODE`，用于冻结门控残差训练默认配置。

## 完整性检查结果

- `artifact_count=4`
- `missing_component_refs=3`：
  - `liquid_btc_tsmixer_v2.pt`
  - `liquid_eth_tsmixer_v2.pt`
  - `liquid_sol_tsmixer_v2.pt`

该结果表示当前 ensemble manifest 引用了 3 个未落地的神经网络权重文件。后续在 `A-2/A-3` 阶段应明确这些组件是“可选占位”还是“必须补齐”。

### 已确认决策（2026-02-19）

- 所有 `.pt` 组件统一在服务器训练阶段产出。
- 当前本地升级重构阶段不以 `.pt` 缺失作为阻塞条件。

## 后续维护方式

1. 每次基线更新后重复执行冻结脚本。
2. 更新本文件中的快照签名与哈希表。
3. 同步更新 `MULTIMODAL_UPGRADE_PLAN_ZH.md` 的任务状态与变更记录。
