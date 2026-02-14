-- 初始化数据库Schema
-- 包含：价格数据、技术指标、正确的训练数据查询

-- 创建价格表（如果不存在）
CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_prices_symbol_timestamp ON prices(symbol, timestamp DESC);

-- 创建技术指标表
CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    ma_7 DECIMAL(20, 8),
    ma_30 DECIMAL(20, 8),
    rsi DECIMAL(5, 2),
    macd DECIMAL(20, 8),
    macd_signal DECIMAL(20, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ti_symbol_timestamp ON technical_indicators(symbol, timestamp);

-- 创建训练样本表（修复后的正确逻辑）
CREATE TABLE IF NOT EXISTS training_samples (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price_start DECIMAL(20, 8) NOT NULL,
    price_end DECIMAL(20, 8) NOT NULL,
    pct_change DECIMAL(10, 4) NOT NULL,
    direction VARCHAR(20) NOT NULL,  -- 'up', 'neutral', 'down'
    embedding JSONB,  -- 新闻embedding
    horizon_hours INT NOT NULL,  -- 预测时间窗口
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_samples_symbol ON training_samples(symbol);
CREATE INDEX IF NOT EXISTS idx_training_samples_created ON training_samples(created_at DESC);

-- 更新 predictions 表，添加实际准确率追踪
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_price DECIMAL(20, 8);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_change_pct DECIMAL(10, 4);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS was_correct BOOLEAN;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP WITH TIME ZONE;

-- 创建视图：最新的技术指标
CREATE OR REPLACE VIEW v_latest_ti AS
SELECT DISTINCT ON (symbol)
    symbol,
    ma_7, ma_30, rsi, macd, macd_signal, timestamp
FROM technical_indicators
ORDER BY symbol, timestamp DESC;

-- 修复后的训练数据查询：从价格表生成标签
-- 这个查询会实际用
CREATE OR REPLACE FUNCTION generate_training_samples(
    p_symbol VARCHAR(20),
    p_days INT DEFAULT 30
) RETURNS VOID AS $$
DECLARE
    sample_cursor CURSOR FOR
        SELECT
            p1.id,
            p1.symbol,
            p1.price as price_start,
            p1.timestamp as time_start,
            p2.price as price_end,
            (p2.price - p1.price) / p1.price * 100 as pct_change,
            CASE
                WHEN (p2.price - p1.price) / p1.price > 0.01 THEN 'up'
                WHEN (p2.price - p1.price) / p1.price < -0.01 THEN 'down'
                ELSE 'neutral'
            END as direction,
            EXTRACT(EPOCH FROM (p2.timestamp - p1.timestamp)) / 3600 as horizon_hours,
            nf.embedding
        FROM prices p1
        JOIN prices p2 ON p1.symbol = p2.symbol AND p2.timestamp > p1.timestamp
        LEFT JOIN nim_features nf ON nf.symbol = p1.symbol
            AND ABS(EXTRACT(EPOCH FROM (nf.created_at - p1.timestamp))) < 3600
        WHERE p1.symbol = p_symbol
            AND p1.timestamp > NOW() - (p_days || ' days')::INTERVAL
            AND p2.timestamp BETWEEN p1.timestamp + INTERVAL '1 hour'
                AND p1.timestamp + INTERVAL '24 hours'
        ORDER BY p1.timestamp DESC
        LIMIT 1000;
BEGIN
    -- 先删除旧数据
    DELETE FROM training_samples
    WHERE symbol = p_symbol;

    -- 插入新数据
    FOR sample IN sample_cursor LOOP
        INSERT INTO training_samples (
            symbol, price_start, price_end, pct_change,
            direction, embedding, horizon_hours
        ) VALUES (
            sample.symbol,
            sample.price_start,
            sample.price_end,
            sample.pct_change,
            sample.direction,
            sample.embedding,
            sample.horizon_hours
        );
    END LOOP;

    RAISE NOTICE 'Generated training samples for %: % rows', p_symbol, ROW_COUNT;
END;
$$ LANGUAGE plpgsql;

-- 授权
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO monitor;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO monitor;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO monitor;

COMMENT ON TABLE training_samples IS '训练样本表：从价格历史生成标签，结合新闻embedding';
COMMENT ON TABLE prices IS '价格数据表';
COMMENT ON TABLE technical_indicators IS '技术指标表';
COMMENT ON FUNCTION generate_training_samples IS '从价格历史生成训练样本，自动标注上涨/下跌/盘整';
