"""event availability timestamps and feature alignment columns

Revision ID: 20260215_0010
Revises: 20260215_0009
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0010"
down_revision = "20260215_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE events
            ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS effective_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS source_latency_ms BIGINT,
            ADD COLUMN IF NOT EXISTS market_scope VARCHAR(16) DEFAULT 'crypto';

        CREATE INDEX IF NOT EXISTS idx_events_available_at
            ON events(available_at DESC);
        CREATE INDEX IF NOT EXISTS idx_events_market_scope_time
            ON events(market_scope, COALESCE(available_at, occurred_at) DESC);

        ALTER TABLE feature_snapshots
            ADD COLUMN IF NOT EXISTS feature_available_at TIMESTAMPTZ;

        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_available
            ON feature_snapshots(target, track, COALESCE(feature_available_at, as_of_ts, as_of) DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_feature_snapshots_available;
        ALTER TABLE feature_snapshots
            DROP COLUMN IF EXISTS feature_available_at;

        DROP INDEX IF EXISTS idx_events_market_scope_time;
        DROP INDEX IF EXISTS idx_events_available_at;
        ALTER TABLE events
            DROP COLUMN IF EXISTS market_scope,
            DROP COLUMN IF EXISTS source_latency_ms,
            DROP COLUMN IF EXISTS effective_at,
            DROP COLUMN IF EXISTS available_at,
            DROP COLUMN IF EXISTS ingested_at,
            DROP COLUMN IF EXISTS published_at;
        """
    )
