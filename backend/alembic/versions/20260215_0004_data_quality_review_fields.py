"""data quality structured review fields

Revision ID: 20260215_0004
Revises: 20260215_0003
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0004"
down_revision = "20260215_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE data_quality_audit
            ADD COLUMN IF NOT EXISTS reviewer TEXT,
            ADD COLUMN IF NOT EXISTS verdict VARCHAR(16),
            ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;

        CREATE INDEX IF NOT EXISTS idx_data_quality_audit_verdict
            ON data_quality_audit(verdict);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_data_quality_audit_verdict;
        ALTER TABLE data_quality_audit
            DROP COLUMN IF EXISTS reviewed_at,
            DROP COLUMN IF EXISTS verdict,
            DROP COLUMN IF EXISTS reviewer;
        """
    )
