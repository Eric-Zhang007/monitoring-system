"""multimodal enrichment side table

Revision ID: 20260219_0012
Revises: 20260215_0011
Create Date: 2026-02-19
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260219_0012"
down_revision = "20260215_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS enriched_event_features (
            id BIGSERIAL PRIMARY KEY,
            event_id BIGINT NOT NULL UNIQUE REFERENCES events(id) ON DELETE CASCADE,
            social_platform VARCHAR(32) NOT NULL DEFAULT 'other',
            social_kind VARCHAR(16) NOT NULL DEFAULT 'post',
            language VARCHAR(16) DEFAULT '',
            summary TEXT DEFAULT '',
            sentiment DOUBLE PRECISION DEFAULT 0.0,
            author TEXT DEFAULT '',
            author_followers BIGINT DEFAULT 0,
            engagement_score DOUBLE PRECISION DEFAULT 0.0,
            embedding JSONB DEFAULT '[]'::jsonb,
            observed_at TIMESTAMPTZ,
            event_time TIMESTAMPTZ,
            ingest_lag_sec DOUBLE PRECISION DEFAULT 0.0,
            coverage_score DOUBLE PRECISION DEFAULT 0.0,
            payload JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_enriched_event_features_observed
            ON enriched_event_features(observed_at DESC);

        CREATE INDEX IF NOT EXISTS idx_enriched_event_features_platform_time
            ON enriched_event_features(social_platform, event_time DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_enriched_event_features_platform_time;
        DROP INDEX IF EXISTS idx_enriched_event_features_observed;
        DROP TABLE IF EXISTS enriched_event_features;
        """
    )

