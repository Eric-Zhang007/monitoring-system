"""add supersede fields to backtest_runs

Revision ID: 20260215_0008
Revises: 20260215_0007
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0008"
down_revision = "20260215_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE backtest_runs
          ADD COLUMN IF NOT EXISTS superseded_by_run_id BIGINT,
          ADD COLUMN IF NOT EXISTS supersede_reason VARCHAR(64),
          ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ;

        CREATE INDEX IF NOT EXISTS idx_backtest_runs_superseded
          ON backtest_runs(track, superseded_by_run_id, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_backtest_runs_superseded;
        ALTER TABLE backtest_runs
          DROP COLUMN IF EXISTS superseded_at,
          DROP COLUMN IF EXISTS supersede_reason,
          DROP COLUMN IF EXISTS superseded_by_run_id;
        """
    )
