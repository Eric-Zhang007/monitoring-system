# Liquid 基线冻结记录（2026-02-19）

## 说明

1. 本文件是历史快照，作用是追溯当时冻结状态。
2. 快照中的维度与工件情况不代表当前代码真实状态。
3. 当前真实状态请看：`docs/STRICT_PIPELINE_ZH.md`。

## 冻结命令

```bash
python3 scripts/freeze_liquid_baseline.py \
  --out baseline_snapshots/liquid_baseline_freeze_20260219T040000Z.json
```

## 冻结产物

1. `baseline_snapshots/liquid_baseline_freeze_20260219T040000Z.json`
2. `baseline_snapshots/liquid_baseline_latest.json`
3. 快照签名：`b067ba3ecf5e478c4b15b9a31b001eddf9c42424d21ecef1c16c87d751ec6819`

## 冻结时模型工件哈希

| 文件 | SHA256 |
|---|---|
| `backend/models/liquid_btc_lgbm_baseline_v2.json` | `669641f592c89d978fc6216a71b66d07b1286e12096bc898306a91c99049fdbb` |
| `backend/models/liquid_eth_lgbm_baseline_v2.json` | `a56c1e1b6c5ab717630f55708cb74d3b34824bd27b2c2bcd1c5ccf893c9e0acd` |
| `backend/models/liquid_sol_lgbm_baseline_v2.json` | `49913173b3015bde0916855fe53edd1486b361697750fa8bb3f00652eba2bafe` |
| `backend/models/liquid_ttm_ensemble_v2.1.json` | `2bc1c7735f1473c6cb43eeaa217624cc0b49b5221ac016ae5331168f360eb4ac` |

## 快照阶段已知问题

1. ensemble manifest 引用的部分 `.pt` 组件当时未落地。
2. 因此该冻结只代表“当时可追溯配置”，不代表“当时工件完整”。
