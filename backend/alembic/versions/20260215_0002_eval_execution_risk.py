"""phase0 eval/execution/risk schema

Revision ID: 20260215_0002
Revises: 20260214_0001
Create Date: 2026-02-15
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260215_0002"
down_revision = "20260214_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE events
            ADD COLUMN IF NOT EXISTS event_importance DOUBLE PRECISION DEFAULT 0.5,
            ADD COLUMN IF NOT EXISTS novelty_score DOUBLE PRECISION DEFAULT 0.5,
            ADD COLUMN IF NOT EXISTS entity_confidence DOUBLE PRECISION DEFAULT 0.5,
            ADD COLUMN IF NOT EXISTS latency_ms BIGINT,
            ADD COLUMN IF NOT EXISTS dedup_cluster_id TEXT;

        ALTER TABLE predictions_v2
            ADD COLUMN IF NOT EXISTS horizon VARCHAR(16),
            ADD COLUMN IF NOT EXISTS model_id BIGINT,
            ADD COLUMN IF NOT EXISTS feature_set_id VARCHAR(128),
            ADD COLUMN IF NOT EXISTS decision_id VARCHAR(64);

        CREATE INDEX IF NOT EXISTS idx_predictions_v2_decision_id
            ON predictions_v2(decision_id);

        CREATE TABLE IF NOT EXISTS signal_candidates (
            id BIGSERIAL PRIMARY KEY,
            track VARCHAR(16) NOT NULL,
            target TEXT NOT NULL,
            horizon VARCHAR(16) NOT NULL,
            score DOUBLE PRECISION NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            action VARCHAR(16) NOT NULL,
            policy VARCHAR(64) NOT NULL,
            decision_id VARCHAR(64) NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_signal_candidates_track_target_time
            ON signal_candidates(track, target, created_at DESC);

        CREATE TABLE IF NOT EXISTS orders_sim (
            id BIGSERIAL PRIMARY KEY,
            decision_id VARCHAR(64) NOT NULL,
            target TEXT NOT NULL,
            track VARCHAR(16) NOT NULL,
            side VARCHAR(8) NOT NULL,
            quantity DOUBLE PRECISION NOT NULL,
            est_price DOUBLE PRECISION,
            est_cost_bps DOUBLE PRECISION DEFAULT 0.0,
            status VARCHAR(32) DEFAULT 'simulated',
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_orders_sim_decision_id
            ON orders_sim(decision_id);

        CREATE TABLE IF NOT EXISTS positions_snapshots (
            id BIGSERIAL PRIMARY KEY,
            decision_id VARCHAR(64) NOT NULL,
            target TEXT NOT NULL,
            track VARCHAR(16) NOT NULL,
            weight DOUBLE PRECISION NOT NULL,
            reason VARCHAR(64) DEFAULT 'rebalance',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_positions_snapshots_decision_id
            ON positions_snapshots(decision_id);

        CREATE TABLE IF NOT EXISTS risk_events (
            id BIGSERIAL PRIMARY KEY,
            decision_id VARCHAR(64),
            severity VARCHAR(16) NOT NULL,
            code VARCHAR(64) NOT NULL,
            message TEXT NOT NULL,
            payload JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_risk_events_severity_time
            ON risk_events(severity, created_at DESC);

        CREATE TABLE IF NOT EXISTS model_promotions (
            id BIGSERIAL PRIMARY KEY,
            track VARCHAR(16) NOT NULL,
            model_name VARCHAR(128) NOT NULL,
            model_version VARCHAR(64) NOT NULL,
            passed BOOLEAN NOT NULL,
            metrics JSONB DEFAULT '{}'::jsonb,
            gate_reason TEXT,
            promoted_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_promotions_track_name_version
            ON model_promotions(track, model_name, model_version);

        CREATE TABLE IF NOT EXISTS data_quality_audit (
            id BIGSERIAL PRIMARY KEY,
            event_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
            source_name TEXT,
            freshness_sec BIGINT,
            entity_coverage DOUBLE PRECISION,
            quality_score DOUBLE PRECISION,
            notes TEXT,
            audited_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_data_quality_audit_event_id
            ON data_quality_audit(event_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS data_quality_audit;
        DROP TABLE IF EXISTS model_promotions;
        DROP TABLE IF EXISTS risk_events;
        DROP TABLE IF EXISTS positions_snapshots;
        DROP TABLE IF EXISTS orders_sim;
        DROP TABLE IF EXISTS signal_candidates;

        DROP INDEX IF EXISTS idx_predictions_v2_decision_id;
        ALTER TABLE predictions_v2
            DROP COLUMN IF EXISTS decision_id,
            DROP COLUMN IF EXISTS feature_set_id,
            DROP COLUMN IF EXISTS model_id,
            DROP COLUMN IF EXISTS horizon;

        ALTER TABLE events
            DROP COLUMN IF EXISTS dedup_cluster_id,
            DROP COLUMN IF EXISTS latency_ms,
            DROP COLUMN IF EXISTS entity_confidence,
            DROP COLUMN IF EXISTS novelty_score,
            DROP COLUMN IF EXISTS event_importance;
        """
    )
