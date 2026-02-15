"""model state for promotion/rollback

Revision ID: 20260215_0003
Revises: 20260215_0002
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0003"
down_revision = "20260215_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS active_model_state (
            track VARCHAR(16) PRIMARY KEY,
            active_model_name VARCHAR(128) NOT NULL,
            active_model_version VARCHAR(64) NOT NULL,
            previous_model_name VARCHAR(128),
            previous_model_version VARCHAR(64),
            status VARCHAR(32) NOT NULL DEFAULT 'active',
            metadata JSONB DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_active_model_state_updated_at
            ON active_model_state(updated_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS active_model_state;
        """
    )
