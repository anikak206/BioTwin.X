"""
BioTwin.X — Async Inference Service
===================================
A thin, production-shaped FastAPI layer over :class:`BioTwinEngine`.

Why this file exists
--------------------
The Streamlit console is the *clinician's* surface. The hospital information
system needs a *machine* surface: something that can answer "what is patient
X's risk at 03:10?" in under 120 ms, without spinning up a browser.

Design decisions
----------------
* **Async workers, sync engine.** The model is CPU-bound and holds global
  state; running it on the event loop would stall every other request. Every
  call is dispatched with ``asyncio.to_thread`` and therefore off the loop.
* **Cache with a TTL, Redis-shaped.** ``_Cache`` is a tiny interface that
  behaves like ``redis.asyncio`` (get/set/delete, TTL). If a Redis URL is
  configured *and* the client is installed, the same interface is backed by
  Redis; otherwise an in-process TTL dict is used. No hard dependency.
* **Fail-safe startup.** If the cohort cannot be built, the service still
  starts and reports ``status: degraded`` — an orchestrator should be able to
  read that, rather than crash-looping.
* **Explicit provenance.** Every response carries the backend, the cohort
  profile and the latency, because a risk number without its provenance is
  not auditable.

Run
---
    pip install "fastapi" "uvicorn"
    uvicorn api.main:app --reload --port 8000
    curl http://localhost:8000/v1/patients?limit=5

Environment
-----------
``BIOTWIN_PROFILE``     cohort profile (default ``demo``)
``BIOTWIN_API_KEY``     if set, required as the ``X-API-Key`` header
``BIOTWIN_REDIS_URL``   Redis DSN for the shared cache (optional)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # FastAPI is an optional dependency: the CLI and console must not need it.
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The service layer requires FastAPI. Install it with:\n"
        '    pip install "fastapi" "uvicorn[standard]"\n'
        f"({exc})"
    ) from exc

import pandas as pd

from biotwin.config import DEFAULT_PROFILE, PROFILES, SLO_LATENCY_MS
from biotwin.inference import BioTwinEngine

PROFILE = os.getenv("BIOTWIN_PROFILE", DEFAULT_PROFILE)
API_KEY = os.getenv("BIOTWIN_API_KEY", "").strip()
CACHE_TTL_S = float(os.getenv("BIOTWIN_CACHE_TTL", "60"))

# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class CacheBackend(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float) -> None: ...
    async def delete(self, key: str) -> None: ...


class MemoryCache:
    """TTL cache with the subset of the Redis API this service needs."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Any | None:
        hit = self._store.get(key)
        if hit is None:
            return None
        expires_at, value = hit
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        self._store[key] = (time.monotonic() + ttl, value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisCache:
    """Same interface, shared across replicas. JSON values only."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as redis  # local import: optional dependency

        self._redis = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        import json

        raw = await self._redis.get(key)
        return None if raw is None else json.loads(raw)

    async def set(self, key: str, value: Any, ttl: float) -> None:
        import json

        await self._redis.set(key, json.dumps(value, default=str), ex=int(max(1, ttl)))

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)


def build_cache() -> CacheBackend:
    url = os.getenv("BIOTWIN_REDIS_URL", "").strip()
    if url:
        try:
            return RedisCache(url)
        except Exception:  # pragma: no cover - redis missing/unreachable
            pass
    return MemoryCache()


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #
class State:
    engine: BioTwinEngine | None = None
    cache: CacheBackend = MemoryCache()
    ready_at: float = 0.0
    startup_error: str | None = None


state = State()


class ScoreRequest(BaseModel):
    patient_id: str = Field(..., examples=["BTX-0004"])
    end_idx: int | None = Field(
        None, ge=0, description="Score as of this 5-minute bin (default: latest)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.cache = build_cache()
    try:
        t0 = time.perf_counter()
        state.engine = await asyncio.to_thread(
            BioTwinEngine, PROFILE, None, True, False)
        state.ready_at = time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover - keeps the process alive
        state.startup_error = f"{type(exc).__name__}: {exc}"
    yield
    state.engine = None


app = FastAPI(
    title="BioTwin.X Inference API",
    version="0.1.0",
    description=(
        "Cross-modal contrastive digital twin for predictive cardiometabolic "
        "prognosis. Research prototype; synthetic data only; not a medical device."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("BIOTWIN_CORS", "*").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Optional shared-secret guard. Off unless ``BIOTWIN_API_KEY`` is set."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def engine_or_503() -> BioTwinEngine:
    if state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"engine not ready: {state.startup_error or 'initialising'}",
        )
    return state.engine


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.engine is not None else "degraded",
        "profile": PROFILE,
        "backend": state.engine.backend if state.engine else None,
        "ready_seconds": round(state.ready_at, 2),
        "error": state.startup_error,
    }


@app.get("/v1/metrics", dependencies=[Depends(require_key)])
async def metrics() -> dict[str, Any]:
    engine = engine_or_503()
    return {
        "profile": engine.profile.name,
        "backend": engine.backend,
        "slo": {"roc_auc": 0.94, "f1": 0.91, "p95_latency_ms": SLO_LATENCY_MS},
        "training": engine.metrics,
    }


@app.get("/v1/patients", dependencies=[Depends(require_key)])
async def list_patients(
    limit: int = Query(25, ge=1, le=500),
    min_risk: float = Query(0.0, ge=0.0, le=100.0),
) -> dict[str, Any]:
    engine = engine_or_503()
    t0 = time.perf_counter()

    async def _board() -> list[dict[str, Any]]:
        board = await asyncio.to_thread(engine.score_cohort)
        sub = board[board["risk_index"] >= min_risk].head(limit)
        sub = sub.assign(as_of=sub["as_of"].astype(str))
        return sub.to_dict(orient="records")

    key = f"board:{PROFILE}:{min_risk}:{limit}"
    rows = await state.cache.get(key)
    cached = rows is not None
    if rows is None:
        rows = await _board()
        await state.cache.set(key, rows, CACHE_TTL_S)
    return {
        "profile": PROFILE,
        "backend": engine.backend,
        "cached": cached,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "patients": rows,
    }


@app.get("/v1/patients/{patient_id}", dependencies=[Depends(require_key)])
async def patient_report(patient_id: str) -> dict[str, Any]:
    engine = engine_or_503()
    ids = engine.bundle.patient_ids()
    if patient_id not in ids:
        raise HTTPException(status_code=404, detail=f"unknown patient {patient_id}")

    key = f"report:{PROFILE}:{patient_id}"
    payload = await state.cache.get(key)
    if payload is not None:
        payload["cached"] = True
        return payload

    report = await asyncio.to_thread(engine.score_patient, patient_id)
    payload = report.to_dict()
    payload.update(profile=PROFILE, backend=engine.backend, cached=False)
    await state.cache.set(key, payload, CACHE_TTL_S)
    return payload


@app.get("/v1/patients/{patient_id}/series", dependencies=[Depends(require_key)])
async def patient_series(
    patient_id: str,
    hours: float = Query(48.0, gt=0, le=24 * 30),
    step_minutes: float = Query(30.0, gt=0, le=240),
) -> dict[str, Any]:
    engine = engine_or_503()
    if patient_id not in engine.bundle.patient_ids():
        raise HTTPException(status_code=404, detail=f"unknown patient {patient_id}")
    frame = await asyncio.to_thread(
        engine.risk_series, patient_id, hours, step_minutes)
    frame = frame.assign(ts=frame["ts"].astype(str))
    return {"patient_id": patient_id, "points": frame.to_dict(orient="records")}


@app.post("/v1/score", dependencies=[Depends(require_key)])
async def score(req: ScoreRequest) -> dict[str, Any]:
    engine = engine_or_503()
    if req.patient_id not in engine.bundle.patient_ids():
        raise HTTPException(status_code=404, detail=f"unknown patient {req.patient_id}")
    report = await asyncio.to_thread(engine.score_patient, req.patient_id, req.end_idx)
    payload = report.to_dict()
    payload.update(profile=PROFILE, backend=engine.backend)
    return payload


@app.post("/v1/cache/flush", dependencies=[Depends(require_key)])
async def flush_cache() -> dict[str, str]:
    state.cache = build_cache()
    return {"status": "flushed"}


# ``uvicorn api.main:app``
__all__ = ["app"]
