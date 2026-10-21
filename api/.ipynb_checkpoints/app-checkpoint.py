"""FastAPI scoring service.   uvicorn api.app:app --host 0.0.0.0 --port 8000

POST /score        one customer: {"customer_id": "C1", "features": {...}}  (missing features are scored as null)
POST /score/batch  many customers
GET  /health, GET /model
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving import ScoringService

app = FastAPI(title="Telco Credit Scoring API", version="1.0")
_service: ScoringService | None = None


def service() -> ScoringService:
    global _service
    if _service is None:
        try:
            _service = ScoringService()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return _service


class ScoreRequest(BaseModel):
    customer_id: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    items: list[ScoreRequest]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model():
    return service().info()


@app.post("/score")
def score(req: ScoreRequest):
    out = service().score(req.features)
    return {"customer_id": req.customer_id, **out}


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    outs = service().score_batch([i.features for i in req.items])
    return [{"customer_id": i.customer_id, **o} for i, o in zip(req.items, outs)]
