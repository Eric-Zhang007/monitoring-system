"""rollout state and data-quality review logs

Revision ID: 20260215_0007
Revises: 20260215_0006
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0007"
down_revision = "20260215_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
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

        CREATE INDEX IF NOT EXISTS idx_dq_review_logs_event_time
            ON data_quality_review_logs(event_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_dq_review_logs_reviewer_time
            ON data_quality_review_logs(reviewer, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_dq_review_logs_reviewer_time;
        DROP INDEX IF EXISTS idx_dq_review_logs_event_time;
        DROP TABLE IF EXISTS data_quality_review_logs;
        DROP TABLE IF EXISTS model_rollout_state;
        """
    )
