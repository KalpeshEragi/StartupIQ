"""Market outlook API: the sector list and one startup's market forecast with "you vs your market"."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import Access, startup_access
from app.core.db import get_db
from app.finance.pnl import build_months
from app.market import service

router = APIRouter(tags=["market"])


@router.get("/market/sectors")
def sectors() -> list[dict]:
    return [{"key": k, "label": label} for k, (label, _) in service.SECTORS.items()]


@router.get("/startups/{startup_id}/market")
def market(access: Access = Depends(startup_access()), db: Session = Depends(get_db)) -> dict:
    startup = access.startup
    suggested = service.suggest(startup.sector)
    key = startup.market_sector or suggested
    data = service.load()
    base = {"sector": key, "chosen": startup.market_sector, "suggested": suggested,
            "available": data is not None and key in (data or {}).get("sectors", {})}
    if data is None:
        return {**base, "message": "The market forecast has not been generated yet (run: python -m market.forecast in ml/)."}
    if not base["available"]:
        return {**base, "message": "Pick the market your startup sells into to see its outlook."}
    outlook = service.outlook(data, key)
    revenue = {m.month.strftime("%Y-%m"): m.revenue for m in build_months(db, startup.id)}
    return {**base, "generated_at": data["generated_at"], "data_through": data["data_through"], "about": data["model"],
            "outlook": outlook, "you_vs_market": service.you_vs_market(revenue, outlook["monthly"])}
