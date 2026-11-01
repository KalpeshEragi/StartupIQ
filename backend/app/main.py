"""Startup IQ API.  Run locally:  uvicorn app.main:app --reload  (from backend/)."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes import auth, equity, financials, market, startups, transactions, trust
from app.core.config import get_settings
from app.core.db import engine

settings = get_settings()
app = FastAPI(title="Startup IQ API", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                   allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type"])

app.include_router(auth.router, prefix="/api/v1")
app.include_router(startups.router, prefix="/api/v1")
app.include_router(transactions.router, prefix="/api/v1")
app.include_router(financials.router, prefix="/api/v1")
app.include_router(equity.router, prefix="/api/v1")
app.include_router(market.router, prefix="/api/v1")
app.include_router(trust.router, prefix="/api/v1")


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "version": app.version}
