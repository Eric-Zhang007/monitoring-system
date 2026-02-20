"""ops control state source-of-truth

Revision ID: 20260219_0013
Revises: 20260219_0012
Create Date: 2026-02-19
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260219_0013"
down_revision = "20260219_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ops_control_state (
            control_key VARCHAR(64) PRIMARY KEY,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            source VARCHAR(32) NOT NULL DEFAULT 'api',
            updated_by VARCHAR(128) NOT NULL DEFAULT 'system',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_ops_control_state_updated_at
            ON ops_control_state(updated_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_ops_control_state_updated_at;
        DROP TABLE IF EXISTS ops_control_state;
        """
    )

