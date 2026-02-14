# 代码追踪与问题清单

## 🔴 已确认的关键问题

### ✅ 已修复的问题

#### 1. 训练数据逻辑错误（training/main.py）
**问题：** 训练时从 predictions 表获取标签，但 predictions 还不存在
```python
# 旧错误代码
query = """
    SELECT nf.embedding, p.direction
    FROM nim_features nf
    LEFT JOIN predictions p ON ...
"""
```
**修复：** ✅ 从价格表直接生成标签（上涨/下跌/盘整）
```python
# 新正确代码
query = """
    WITH price_windows AS (
        SELECT
            p1.price as price_start,
            p2.price as price_end,
            (p2.price - p1.price) / p1.price * 100 as pct_change,
            CASE
                WHEN (p2.price - p1.price) / p1.price > 0.5 THEN 'up'
                WHEN (p2.price - p1.price) / p1.price < -0.5 THEN 'down'
                ELSE 'neutral'
            END as direction
        FROM prices p1
        JOIN prices p2 ON ...
    )
    SELECT direction, embedding FROM price_windows
"""
```

#### 2. 验证函数造假（training/main.py）
**问题：** 验证指标是硬编码的假数据
```python
# 旧错误代码
def train_with_validation(self, val_features, val_labels):
    return {
        "train_accuracy": 0.75,  # 假数据！
    }
```
**修复：** ✅ 真实的训练循环，计算真实准确率和损失
```python
# 新正确代码
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == batch_labels).sum().item()

    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            ...
            val_correct += (predicted == batch_labels).sum().item()

    # 真实记录指标
    train_accuracy = 100 * train_correct / train_total
    val_accuracy = 100 * val_correct / val_total
```

#### 3. 推理使用随机权重（inference/main.py）
**问题：** 预测时使用随机初始化的权重
```python
# 旧错误代码
scenario_logits = torch.nn.functional.linear(
    last_output,
    torch.randn(3, 256).to(self.device)  # 随机！
)
```
**修复：** ✅ 加载训练好的模型权重
```python
# 新正确代码
def load_model(self, symbol: str):
    model_path = os.path.join(MODEL_DIR, f"{symbol.lower()}_model.pth")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        # 创建模型
        self.model = ImprovedModel(**checkpoint['model_config']).to(self.device)

        # 加载真实权重（修复点）
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"✅ Loaded model for {symbol}")
```

#### 4. 后端API全用Mock数据（backend/main.py）
**问题：** 所有API端点都返回硬编码的假数据
```python
# 旧错误代码
@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str):
    return {
        "symbol": symbol,
        "predictions": [  # 硬编码假数据
            {"horizon": "1h", "direction": "up", "change": "+1.2%", ...},
        ]
    }
```
**修复：** ✅ 从数据库查询真实数据
```python
# 新正确代码
@app.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str, hours: int = 24):
    try:
        conn = get_postgres()
        cursor = conn.cursor()

        query = """
            SELECT
                symbol, scenario, direction, confidence,
                expected_change_pct, expected_price,
                scenario_probabilities, created_at
            FROM predictions
            WHERE symbol = UPPER(%s)
              AND created_at > NOW() - make_interval(hours => %s)
            ORDER BY created_at DESC
            LIMIT 100
        """

        cursor.execute(query, (symbol, hours))
        rows = cursor.fetchall()

        predictions = [dict(row) for row in rows]
        return {"symbol": symbol.upper(), "predictions": predictions}
    except Exception as e:
        logger.error(f"❌ Failed to get predictions: {e}")
```

#### 5. 数据库Schema不完整（新增）
**问题：** 缺少价格表、技术指标表、正确的训练样本表
**修复：** ✅ 创建了 `scripts/init_db.sql`
```sql
-- 价格表
CREATE TABLE prices (...);
-- 技术指标表
CREATE TABLE technical_indicators (...);
-- 训练样本表
CREATE TABLE training_samples (...);
-- 修复后的训练数据查询函数
CREATE OR REPLACE FUNCTION generate_training_samples(...);
```

---

## 🎯 修复进度

### P0 - 立即修复 ✅ 全部完成
- [x] ✅ 修复training数据逻辑（从价格表生成标签）
- [x] ✅ 实现真实的模型加载（推理服务）
- [x] ✅ 修复后端Mock数据（真实数据库查询）
- [x] ✅ 添加数据库Schema（价格、技术指标、训练样本）
- [x] ✅ 添加真实验证逻辑（准确率、损失）

### P1 - 高优先级
- [x] ✅ 添加价格采集支持（Schema支持）
- [x] ✅ 添加技术指标Schema
- [ ] ⏳ 实现价格数据采集（collector.py）
- [ ] ⏳ 实现技术指标计算

### P2 - 中优先级
- [ ] ⏳ 特征工程优化（多时序窗口）
- [ ] ⏳ 模型架构优化（Transformer/TCN/TFT）
- [ ] ⏳ 评估指标完善（Sharpe Ratio, Max Drawdown）

---

## 📝 当前状态

### 已完成 ✅
1. **Training Service修复：**
   - 从价格表生成正确的训练标签
   - 真实的训练/验证循环
   - 模型保存和加载

2. **Inference Service修复：**
   - 真实加载训练好的模型
   - 不再使用随机权重
   - 正确的特征查询逻辑

3. **Backend API修复：**
   - 所有Mock数据已移除
   - 从PostgreSQL查询真实预测
   - 从Redis缓存价格数据
   - WebSocket推送真实数据

4. **数据库Schema：**
   - 价格表（prices）
   - 技术指标表（technical_indicators）
   - 训练样本表（training_samples）
   - 修复的SQL函数

### 待完成 ⏳
1. 数据采集：collector.py需要实现真实的价格采集
2. 技术指标：需要实现MA、MACD、RSI的计算
3. 端到端测试：在有GPU的环境下运行完整测试
4. 性能优化：模型量化、batch推理等

---

## 🔄 Git提交历史

### Commit 1: MVP Complete (7ccd359)
- 初始MVP版本
- 包含架构和基础代码
- 但有上述问题

### Commit 2: Fix Critical Issues (待提交)
- 修复训练数据逻辑
- 修复推理权重加载
- 修复后端Mock数据
- 添加数据库Schema

---

**修复时间：** 2026-02-14 23:15 - 23:30
**修复者：** 小黑
**状态：** ✅ P0问题全部修复，待推送到GitHub
