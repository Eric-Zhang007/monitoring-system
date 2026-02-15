"""add run_source field to backtest_runs

Revision ID: 20260215_0009
Revises: 20260215_0008
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0009"
down_revision = "20260215_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE backtest_runs
          ADD COLUMN IF NOT EXISTS run_source VARCHAR(32) NOT NULL DEFAULT 'prod';

        CREATE INDEX IF NOT EXISTS idx_backtest_runs_track_source_created
          ON backtest_runs(track, run_source, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_backtest_runs_track_source_created;
        ALTER TABLE backtest_runs
          DROP COLUMN IF EXISTS run_source;
        """
    )
