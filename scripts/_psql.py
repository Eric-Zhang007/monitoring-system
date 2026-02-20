#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess


def run_psql(sql: str) -> str:
    dsn = str(os.getenv("METRICS_DATABASE_URL", "")).strip() or str(os.getenv("DATABASE_URL", "")).strip()
    if not dsn:
        raise RuntimeError("DATABASE_URL (or METRICS_DATABASE_URL) is required")
    cmd = ["psql", dsn, "-At", "-F", "|", "-c", sql]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0:
        return p.stdout.strip()
    raise RuntimeError(p.stderr.strip() or "psql execution failed")
