"""risk control state table

Revision ID: 20260215_0006
Revises: 20260215_0005
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0006"
down_revision = "20260215_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
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

        CREATE INDEX IF NOT EXISTS idx_risk_control_state_track_state
            ON risk_control_state(track, state, updated_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_risk_control_state_track_state;
        DROP TABLE IF EXISTS risk_control_state;
        """
    )
