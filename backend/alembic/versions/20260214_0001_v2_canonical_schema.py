"""v2 canonical schema

Revision ID: 20260214_0001
Revises:
Create Date: 2026-02-14
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260214_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id BIGSERIAL PRIMARY KEY,
            entity_type VARCHAR(32) NOT NULL,
            name TEXT NOT NULL,
            symbol VARCHAR(20),
            country VARCHAR(8),
            sector TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(entity_type, name)
        );

        CREATE TABLE IF NOT EXISTS events (
            id BIGSERIAL PRIMARY KEY,
            event_type VARCHAR(32) NOT NULL,
            title TEXT NOT NULL,
            occurred_at TIMESTAMPTZ NOT NULL,
            source_url TEXT,
            source_name TEXT,
            source_timezone VARCHAR(64) DEFAULT 'UTC',
            source_tier SMALLINT DEFAULT 3,
            confidence_score DOUBLE PRECISION DEFAULT 0.5,
            payload JSONB DEFAULT '{}'::jsonb,
            fingerprint VARCHAR(64) UNIQUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

        CREATE TABLE IF NOT EXISTS event_links (
            id BIGSERIAL PRIMARY KEY,
            event_id BIGINT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            role VARCHAR(64) DEFAULT 'mentioned',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(event_id, entity_id, role)
        );

        CREATE TABLE IF NOT EXISTS feature_snapshots (
            id BIGSERIAL PRIMARY KEY,
            target TEXT NOT NULL,
            track VARCHAR(16) NOT NULL,
            as_of TIMESTAMPTZ NOT NULL,
            feature_version VARCHAR(64) NOT NULL,
            feature_payload JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_target_track_asof
            ON feature_snapshots(target, track, as_of DESC);

        CREATE TABLE IF NOT EXISTS model_registry (
            id BIGSERIAL PRIMARY KEY,
            model_name VARCHAR(128) NOT NULL,
            track VARCHAR(16) NOT NULL,
            model_version VARCHAR(64) NOT NULL,
            artifact_path TEXT NOT NULL,
            metrics JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_name_track_version
            ON model_registry(model_name, track, model_version);

        CREATE TABLE IF NOT EXISTS predictions_v2 (
            id BIGSERIAL PRIMARY KEY,
            track VARCHAR(16) NOT NULL,
            target TEXT NOT NULL,
            score DOUBLE PRECISION NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            outputs JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_v2_track_target_time
            ON predictions_v2(track, target, created_at DESC);

        CREATE TABLE IF NOT EXISTS prediction_explanations (
            id BIGSERIAL PRIMARY KEY,
            prediction_id BIGINT NOT NULL REFERENCES predictions_v2(id) ON DELETE CASCADE,
            top_event_contributors JSONB DEFAULT '[]'::jsonb,
            top_feature_contributors JSONB DEFAULT '[]'::jsonb,
            evidence_links JSONB DEFAULT '[]'::jsonb,
            model_version VARCHAR(64) NOT NULL,
            feature_version VARCHAR(64) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_prediction_explanations_prediction
            ON prediction_explanations(prediction_id);

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id BIGSERIAL PRIMARY KEY,
            run_name VARCHAR(128) NOT NULL,
            track VARCHAR(16) NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            ended_at TIMESTAMPTZ,
            metrics JSONB DEFAULT '{}'::jsonb,
            config JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS backtest_runs;
        DROP TABLE IF EXISTS prediction_explanations;
        DROP TABLE IF EXISTS predictions_v2;
        DROP TABLE IF EXISTS model_registry;
        DROP TABLE IF EXISTS feature_snapshots;
        DROP TABLE IF EXISTS event_links;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS entities;
        """
    )
