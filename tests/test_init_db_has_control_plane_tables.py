from __future__ import annotations

from pathlib import Path


def test_init_db_contains_control_plane_tables():
    sql = Path("scripts/init_db.sql").read_text(encoding="utf-8").lower()
    assert "create table if not exists offline_data_audits" in sql
    assert "create table if not exists runtime_config" in sql
    assert "create table if not exists runtime_config_audit_logs" in sql
    assert "create table if not exists ops_processes" in sql
    assert "create table if not exists ops_process_events" in sql
    assert "create table if not exists bitget_accounts" in sql
    assert "create table if not exists risk_command_logs" in sql
    assert "create table if not exists venue_connectivity_status" in sql
    assert "create table if not exists proxy_profiles" in sql
    assert "create table if not exists proxy_profile_bindings" in sql
    assert "create table if not exists audit_logs" in sql
    assert "create table if not exists ops_secrets" in sql
    assert "create table if not exists clock_drift_status" in sql
