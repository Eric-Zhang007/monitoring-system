from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from schemas_v2 import (
    IngestSocialRequest,
    IngestSocialResponse,
    LatestFeatureSnapshotResponse,
    SocialCoverageResponse,
)
from services.multimodal_service import social_items_to_events
from v2_repository import V2Repository


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
router = APIRouter(prefix="/api/v2", tags=["v2-multimodal"])


class _UnavailableRepo:
    def __init__(self, reason: str):
        self._reason = reason

    def __getattr__(self, name: str):
        raise RuntimeError(f"V2Repository unavailable: {self._reason}")


def _init_repo() -> Any:
    if "pytest" in sys.modules and os.getenv("PYTEST_INIT_DB", "0").strip() not in {"1", "true", "yes"}:
        return _UnavailableRepo("skipped repo init during pytest import")
    try:
        return V2Repository(DATABASE_URL)
    except Exception as exc:
        return _UnavailableRepo(str(exc))


repo = _init_repo()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@router.post("/ingest/social", response_model=IngestSocialResponse)
async def ingest_social(payload: IngestSocialRequest) -> IngestSocialResponse:
    if not payload.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")
    events, enrich_rows = social_items_to_events(payload.items)
    accepted, inserted, deduplicated, event_ids = repo.ingest_events(events)
    enriched_written = 0
    for event_id, enrich in zip(event_ids, enrich_rows):
        try:
            if repo.upsert_enriched_event_feature(event_id=event_id, payload=enrich):
                enriched_written += 1
        except Exception:
            continue
    return IngestSocialResponse(
        accepted=accepted,
        inserted=inserted,
        deduplicated=deduplicated,
        enriched_written=enriched_written,
        event_ids=event_ids,
    )


@router.get("/features/latest", response_model=LatestFeatureSnapshotResponse)
async def get_latest_feature(
    target: str = Query(..., min_length=1, max_length=32),
    track: str = Query("liquid", min_length=1, max_length=16),
):
    row = repo.latest_feature_snapshot(target=str(target).upper(), track=str(track).lower())
    if not row:
        raise HTTPException(status_code=404, detail="feature_snapshot_not_found")
    return LatestFeatureSnapshotResponse(
        target=str(row.get("target") or "").upper(),
        track=str(row.get("track") or "liquid"),  # type: ignore[arg-type]
        feature_version=str(row.get("feature_version") or ""),
        data_version=str(row.get("data_version") or "v1"),
        as_of=row.get("as_of"),
        as_of_ts=row.get("as_of_ts"),
        event_time=row.get("event_time"),
        feature_available_at=row.get("feature_available_at"),
        lineage_id=row.get("lineage_id"),
        feature_payload=dict(row.get("feature_payload") or {}),
        created_at=row.get("created_at") or _utcnow(),
    )


@router.get("/dq/social-coverage", response_model=SocialCoverageResponse)
async def get_social_coverage(
    window_hours: int = Query(24, ge=1, le=24 * 30),
    target: Optional[str] = Query(None, min_length=1, max_length=32),
) -> SocialCoverageResponse:
    stats = repo.social_coverage_stats(window_hours=window_hours, target=target)
    return SocialCoverageResponse(
        window_hours=window_hours,
        target=(str(target).upper() if target else None),
        totals=dict(stats.get("totals") or {}),
        by_symbol=list(stats.get("by_symbol") or []),
        generated_at=_utcnow(),
    )


@router.get("/dq/social_coverage", response_model=SocialCoverageResponse)
async def get_social_coverage_compat(
    window: int = Query(24, ge=1, le=24 * 30),
    target: Optional[str] = Query(None, min_length=1, max_length=32),
) -> SocialCoverageResponse:
    return await get_social_coverage(window_hours=window, target=target)
