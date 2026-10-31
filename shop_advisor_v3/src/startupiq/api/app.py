"""Startup IQ web app (Task 9): uvicorn startupiq.api.app:app --reload  -> http://127.0.0.1:8000

GET  /            the single-page UI (static/index.html)
POST /api/analyze upload a CSV/Excel file + optional business details -> the full Report as JSON
GET  /api/samples demo businesses (demo/files) with their cash, cost and margin
GET  /api/health  liveness check
The uploaded file is processed in memory and never written to disk (Task 10 privacy).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from startupiq import intake
from startupiq.contracts import BusinessContext
from startupiq.pipeline import analyze_intake

STATIC = Path(__file__).parent / "static"
DEMO = Path(__file__).resolve().parents[3] / "demo"
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Startup IQ", version="0.1.0")


def _num(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def _samples() -> dict[str, dict]:
    inputs = json.loads((DEMO / "inputs.json").read_text(encoding="utf-8")) if (DEMO / "inputs.json").exists() else {}
    return {f.name: inputs.get(f.name, {}) for f in sorted((DEMO / "files").glob("*.csv")) if (f.name[:2].isdigit() and int(f.name[:2]) <= 12) or f.name.startswith("real_")}  # shop scenarios and real data; 13+ test broken uploads


@app.get("/api/samples")
def samples() -> list[dict]:
    def label(n: str) -> str:
        return "Real online-shop product (unseen by the model)" if n.startswith("real_") else n.rsplit(".", 1)[0].split("_", 1)[1].replace("_", " ").capitalize()
    return [{"file": n, "label": label(n), "context": c} for n, c in _samples().items()]


@app.get("/api/evidence")
def evidence() -> dict:
    """Offline results behind the forecast card: the three deep architectures, and the hybrid that serves."""
    res = DEMO.parent / "docs" / "results"
    val, test = (json.loads((res / f).read_text(encoding="utf-8")) for f in ("dl_weekly.json", "dl_weekly_test.json"))
    first = lambda v: float(str(v).split(" ")[0])  # noqa: E731  "0.686 ± 0.009" -> 0.686
    deep = [{"arch": {"mlp": "MLP", "lstm": "LSTM + attention", "tcn": "1D CNN (TCN)"}[v["arch"]], "params": v["params"],
             "val": first(v["pooled MASE"]), "test": first(t["pooled MASE"]), "stores_test": first(t["Rossmann stores"]),
             "products_test": first(t["UCI products"]), "coverage_test": first(t["P10-P90 coverage"])} for v, t in zip(val["seeds"], test["test"])]
    hybrid = json.loads((res / "dl_hybrid.json").read_text(encoding="utf-8"))["summary"]
    synthetic = {name: json.loads((res / f"{name}.json").read_text(encoding="utf-8")) if (res / f"{name}.json").exists() else None
                 for name in ("category", "dl_category")}
    return {"deep": deep, "auto": {"val": val["step5_bar"]["Auto"], "test": test["step5_test_bar"]["Auto"]}, "hybrid": hybrid, **synthetic}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), sample: str = Form(""), cash: str = Form(""), fixed_costs: str = Form(""), margin_pct: str = Form(""),
                  stock_value: str = Form(""), lead_time_days: str = Form(""), competitors: str = Form(""),
                  rival_cut_prices: str = Form(""), column_map: str = Form("")) -> JSONResponse:
    if sample:
        if sample not in _samples():
            return JSONResponse({"error": "Unknown sample."}, status_code=404)
        data, name = (DEMO / "files" / sample).read_bytes(), sample
    elif file is not None:
        data, name = await file.read(), file.filename or "upload.csv"
    else:
        return JSONResponse({"error": "Choose a file or a sample business."}, status_code=422)
    if len(data) > MAX_BYTES:
        return JSONResponse({"error": "File is larger than 20 MB."}, status_code=413)
    margin = _num(margin_pct)
    lead = _num(lead_time_days)
    ctx = BusinessContext(cash_on_hand=_num(cash), monthly_fixed_costs=_num(fixed_costs),
                          gross_margin=None if margin is None else max(0.0, min(1.0, margin / 100)),
                          stock_value=_num(stock_value), supplier_lead_time_days=None if lead is None else int(lead),
                          competitors=[c.strip() for c in competitors.split(",") if c.strip()],
                          rival_price_cut_seen=True if rival_cut_prices == "yes" else (False if rival_cut_prices == "no" else None))
    overrides = json.loads(column_map) if column_map else None
    got = intake.read_upload(data, name, overrides)
    report = analyze_intake(got, ctx)
    return JSONResponse(json.loads(report.model_dump_json()))
