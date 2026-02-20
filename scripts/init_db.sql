-- 初始化数据库Schema
-- 包含：价格数据、技术指标、正确的训练数据查询
CREATE EXTENSION IF NOT EXISTS vector;

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

-- =============================================
-- V2 Canonical Schema (Dual-track VC + Liquid)
-- =============================================

CREATE TABLE IF NOT EXISTS entities (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(32) NOT NULL,
    name TEXT NOT NULL,
    symbol VARCHAR(20),
    country VARCHAR(8),
    sector TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (entity_type, name)
);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(32) NOT NULL,
    title TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ,
    available_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    source_url TEXT,
    source_name TEXT,
    source_timezone VARCHAR(64) DEFAULT 'UTC',
    source_tier SMALLINT DEFAULT 3,
    confidence_score DOUBLE PRECISION DEFAULT 0.5,
    source_latency_ms BIGINT,
    market_scope VARCHAR(16) DEFAULT 'crypto',
    payload JSONB DEFAULT '{}'::jsonb,
    fingerprint VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_available_at ON events(available_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

CREATE TABLE IF NOT EXISTS event_links (
    id BIGSERIAL PRIMARY KEY,
    event_id BIGINT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    role VARCHAR(64) DEFAULT 'mentioned',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (event_id, entity_id, role)
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    target TEXT NOT NULL,
    track VARCHAR(16) NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    feature_version VARCHAR(64) NOT NULL,
    feature_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_target_track_asof
    ON feature_snapshots(target, track, as_of DESC);

CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(128) NOT NULL,
    track VARCHAR(16) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_name_track_version
    ON model_registry(model_name, track, model_version);

CREATE TABLE IF NOT EXISTS predictions_v2 (
    id BIGSERIAL PRIMARY KEY,
    track VARCHAR(16) NOT NULL,
    target TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    outputs JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_predictions_v2_track_target_time
    ON predictions_v2(track, target, created_at DESC);

CREATE TABLE IF NOT EXISTS prediction_explanations (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES predictions_v2(id) ON DELETE CASCADE,
    top_event_contributors JSONB DEFAULT '[]'::jsonb,
    top_feature_contributors JSONB DEFAULT '[]'::jsonb,
    evidence_links JSONB DEFAULT '[]'::jsonb,
    model_version VARCHAR(64) NOT NULL,
    feature_version VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_prediction_explanations_prediction
    ON prediction_explanations(prediction_id);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    run_name VARCHAR(128) NOT NULL,
    track VARCHAR(16) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    metrics JSONB DEFAULT '{}'::jsonb,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signal_candidates (
    id BIGSERIAL PRIMARY KEY,
    track VARCHAR(16) NOT NULL,
    target TEXT NOT NULL,
    horizon VARCHAR(16) NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    action VARCHAR(16) NOT NULL,
    policy VARCHAR(64) NOT NULL,
    decision_id VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signal_candidates_track_target_time ON signal_candidates(track, target, created_at DESC);

CREATE TABLE IF NOT EXISTS orders_sim (
    id BIGSERIAL PRIMARY KEY,
    decision_id VARCHAR(64) NOT NULL,
    target TEXT NOT NULL,
    track VARCHAR(16) NOT NULL,
    side VARCHAR(8) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    est_price DOUBLE PRECISION,
    est_cost_bps DOUBLE PRECISION DEFAULT 0.0,
    status VARCHAR(32) DEFAULT 'simulated',
    adapter VARCHAR(32) DEFAULT 'paper',
    venue VARCHAR(64) DEFAULT 'coinbase',
    time_in_force VARCHAR(16) DEFAULT 'IOC',
    max_slippage_bps DOUBLE PRECISION DEFAULT 20.0,
    strategy_id VARCHAR(64) DEFAULT 'default-liquid-v1',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_orders_sim_decision_id ON orders_sim(decision_id);

-- Phase1 crypto foundation
CREATE TABLE IF NOT EXISTS market_bars (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(8) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    source VARCHAR(64) DEFAULT 'exchange',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_tf_ts ON market_bars(symbol, timeframe, ts DESC);

CREATE TABLE IF NOT EXISTS orderbook_l2 (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    bid_px DOUBLE PRECISION,
    ask_px DOUBLE PRECISION,
    bid_sz DOUBLE PRECISION,
    ask_sz DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    imbalance DOUBLE PRECISION,
    source VARCHAR(64) DEFAULT 'exchange',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_orderbook_l2_symbol_ts ON orderbook_l2(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS trades_ticks (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    side VARCHAR(8),
    source VARCHAR(64) DEFAULT 'exchange',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trades_ticks_symbol_ts ON trades_ticks(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS funding_rates (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    funding_rate DOUBLE PRECISION NOT NULL,
    next_funding_ts TIMESTAMPTZ,
    source VARCHAR(64) DEFAULT 'exchange',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol_ts ON funding_rates(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS onchain_signals (
    id BIGSERIAL PRIMARY KEY,
    asset_symbol VARCHAR(32) NOT NULL,
    chain VARCHAR(32) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(64) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    source VARCHAR(64) DEFAULT 'onchain',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_onchain_signals_asset_ts ON onchain_signals(asset_symbol, ts DESC);

ALTER TABLE feature_snapshots
    ADD COLUMN IF NOT EXISTS as_of_ts TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS feature_available_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS data_version VARCHAR(64) DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS lineage_id VARCHAR(64);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_lineage ON feature_snapshots(lineage_id);

ALTER TABLE events
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS effective_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS source_latency_ms BIGINT,
    ADD COLUMN IF NOT EXISTS market_scope VARCHAR(16) DEFAULT 'crypto';

CREATE TABLE IF NOT EXISTS risk_control_state (
    id BIGSERIAL PRIMARY KEY,
    track VARCHAR(16) NOT NULL,
    strategy_id VARCHAR(64) NOT NULL DEFAULT 'global',
    state VARCHAR(16) NOT NULL DEFAULT 'armed',
    reason TEXT NOT NULL DEFAULT 'init',
    metadata JSONB DEFAULT '{}'::jsonb,
    triggered_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(track, strategy_id)
);
CREATE INDEX IF NOT EXISTS idx_risk_control_state_track_state ON risk_control_state(track, state, updated_at DESC);

CREATE TABLE IF NOT EXISTS model_rollout_state (
    id BIGSERIAL PRIMARY KEY,
    track VARCHAR(16) NOT NULL UNIQUE,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    stage_pct INTEGER NOT NULL DEFAULT 10,
    status VARCHAR(32) NOT NULL DEFAULT 'shadow',
    hard_limits JSONB DEFAULT '{}'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS data_quality_review_logs (
    id BIGSERIAL PRIMARY KEY,
    audit_id BIGINT NOT NULL,
    event_id BIGINT,
    reviewer VARCHAR(128) NOT NULL,
    verdict VARCHAR(32) NOT NULL,
    note TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dq_review_logs_event_time ON data_quality_review_logs(event_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dq_review_logs_reviewer_time ON data_quality_review_logs(reviewer, created_at DESC);

CREATE TABLE IF NOT EXISTS ops_control_state (
    control_key VARCHAR(64) PRIMARY KEY,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    source VARCHAR(32) NOT NULL DEFAULT 'api',
    updated_by VARCHAR(128) NOT NULL DEFAULT 'system',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ops_control_state_updated_at ON ops_control_state(updated_at DESC);
