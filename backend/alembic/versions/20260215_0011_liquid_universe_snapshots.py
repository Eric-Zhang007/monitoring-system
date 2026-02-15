"""asset universe snapshots for as-of backtest target resolution

Revision ID: 20260215_0011
Revises: 20260215_0010
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0011"
down_revision = "20260215_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_universe_snapshots (
            id BIGSERIAL PRIMARY KEY,
            track VARCHAR(16) NOT NULL,
            as_of TIMESTAMPTZ NOT NULL,
            universe_version VARCHAR(64) NOT NULL DEFAULT 'manual_v1',
            source VARCHAR(64) NOT NULL DEFAULT 'manual',
            symbols_json JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_asset_universe_snapshots_track_asof
            ON asset_universe_snapshots(track, as_of DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_asset_universe_snapshots_track_asof;
        DROP TABLE IF EXISTS asset_universe_snapshots;
        """
    )
