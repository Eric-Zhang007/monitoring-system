"""crypto phase1 foundation schema

Revision ID: 20260215_0005
Revises: 20260215_0004
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0005"
down_revision = "20260215_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
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

        CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_tf_ts
            ON market_bars(symbol, timeframe, ts DESC);

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

        CREATE INDEX IF NOT EXISTS idx_orderbook_l2_symbol_ts
            ON orderbook_l2(symbol, ts DESC);

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

        CREATE INDEX IF NOT EXISTS idx_trades_ticks_symbol_ts
            ON trades_ticks(symbol, ts DESC);

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

        CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol_ts
            ON funding_rates(symbol, ts DESC);

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

        CREATE INDEX IF NOT EXISTS idx_onchain_signals_asset_ts
            ON onchain_signals(asset_symbol, ts DESC);

        ALTER TABLE feature_snapshots
            ADD COLUMN IF NOT EXISTS as_of_ts TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS data_version VARCHAR(64) DEFAULT 'v1',
            ADD COLUMN IF NOT EXISTS lineage_id VARCHAR(64);

        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_lineage
            ON feature_snapshots(lineage_id);

        ALTER TABLE orders_sim
            ADD COLUMN IF NOT EXISTS adapter VARCHAR(32) DEFAULT 'paper',
            ADD COLUMN IF NOT EXISTS venue VARCHAR(64) DEFAULT 'coinbase',
            ADD COLUMN IF NOT EXISTS time_in_force VARCHAR(16) DEFAULT 'IOC',
            ADD COLUMN IF NOT EXISTS max_slippage_bps DOUBLE PRECISION DEFAULT 20.0,
            ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(64) DEFAULT 'default-liquid-v1';

        CREATE INDEX IF NOT EXISTS idx_orders_sim_track_status_time
            ON orders_sim(track, status, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_orders_sim_track_status_time;

        ALTER TABLE orders_sim
            DROP COLUMN IF EXISTS strategy_id,
            DROP COLUMN IF EXISTS max_slippage_bps,
            DROP COLUMN IF EXISTS time_in_force,
            DROP COLUMN IF EXISTS venue,
            DROP COLUMN IF EXISTS adapter;

        DROP INDEX IF EXISTS idx_feature_snapshots_lineage;
        ALTER TABLE feature_snapshots
            DROP COLUMN IF EXISTS lineage_id,
            DROP COLUMN IF EXISTS data_version,
            DROP COLUMN IF EXISTS event_time,
            DROP COLUMN IF EXISTS as_of_ts;

        DROP TABLE IF EXISTS onchain_signals;
        DROP TABLE IF EXISTS funding_rates;
        DROP TABLE IF EXISTS trades_ticks;
        DROP TABLE IF EXISTS orderbook_l2;
        DROP TABLE IF EXISTS market_bars;
        """
    )
