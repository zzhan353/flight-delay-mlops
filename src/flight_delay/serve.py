"""Prediction service.

Design notes that matter for the hosting budget and for honesty:

* **The model is baked into the image, not fetched at startup.** Container Apps scales
  this service to zero, so every cold start pays the model-load cost. Pulling the
  artifact from a registry at boot would add seconds and a network failure mode to a
  path that runs on every request after an idle period.

* **Weather features are optional and default to missing.** A traveller asking about a
  future flight has no observed weather, and the model was trained with 36% of rows
  missing the weather join, so `NaN` is a value it has genuinely seen. The `/api/predict`
  contract accepts weather when a caller has it; the demo UI does not send any. See
  README "Known limitations" for the measured cost of that.

* **The response says what the number means.** The model is modestly better than a
  historical-rate lookup, and presenting a bare probability as travel advice would
  overstate it. Every response carries the base rate for comparison and the model's
  own measured quality.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Annotated

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Resolve paths from the package location, never the process working directory: the
# service must behave identically whether it is started by uvicorn from the repo root,
# by pytest, or by the container entrypoint from /.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models" / "candidate"))
STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


def load_reference() -> dict:
    with resources.files("flight_delay.assets").joinpath("reference.json").open() as fh:
        return json.load(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    state["reference"] = load_reference()
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    metrics_path = MODEL_DIR.parent / "candidate_metrics.json"
    state["metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    state["loaded_in_ms"] = round((time.perf_counter() - started) * 1000, 1)
    state["requests"] = 0
    log.info("model ready in %.1f ms", state["loaded_in_ms"])
    yield
    state.clear()


app = FastAPI(
    title="Flight Delay Prediction",
    description="Probability that a US domestic flight arrives 15+ minutes late.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    carrier: Annotated[str, Field(min_length=2, max_length=3, examples=["AA"])]
    origin: Annotated[str, Field(min_length=3, max_length=4, examples=["JFK"])]
    dest: Annotated[str, Field(min_length=3, max_length=4, examples=["LAX"])]
    flight_date: date
    dep_hour: Annotated[int, Field(ge=0, le=23, examples=[17])]
    # Optional weather at the origin, in metric units, if the caller has a forecast.
    prcp: float | None = None
    snow: float | None = None
    tmax: float | None = None
    tmin: float | None = None
    awnd: float | None = None


class PredictResponse(BaseModel):
    delay_probability: float
    base_rate: float
    relative_risk: float
    band: str
    route: str
    model_quality: dict


def _band(prob: float, base: float) -> str:
    """Describe the prediction relative to the base rate rather than in absolute terms.

    A raw "24%" reads as low risk to most people even when it is well above average,
    so the label is anchored to how this flight compares with a typical one.
    """
    ratio = prob / base if base else 1.0
    if ratio < 0.8:
        return "below average"
    if ratio < 1.2:
        return "about average"
    if ratio < 1.6:
        return "above average"
    return "well above average"


def build_frame(req: PredictRequest, reference: dict) -> pd.DataFrame:
    route_key = f"{req.origin.upper()}-{req.dest.upper()}"
    route = reference["routes"].get(route_key)
    if route is None:
        raise HTTPException(
            status_code=404,
            detail=f"No scheduled service on {route_key} in the training data. "
            "The model has never seen this route, so a prediction would be guesswork.",
        )
    distance, elapsed, arr_offset = route
    arr_hour = (req.dep_hour + max(arr_offset, 0) + int(elapsed // 60)) % 24

    return pd.DataFrame(
        [
            {
                "FlightDate": pd.Timestamp(req.flight_date),
                "Month": req.flight_date.month,
                "DayOfWeek": req.flight_date.isoweekday(),
                "DayofMonth": req.flight_date.day,
                "Reporting_Airline": req.carrier.upper(),
                "Origin": req.origin.upper(),
                "Dest": req.dest.upper(),
                "DepTimeBlk": f"{req.dep_hour:02d}00-{req.dep_hour:02d}59",
                "CRSDepHour": req.dep_hour,
                "CRSArrHour": arr_hour,
                "Distance": float(distance),
                "CRSElapsedTime": float(elapsed),
                "PRCP": req.prcp,
                "SNOW": req.snow,
                "TMAX": req.tmax,
                "TMIN": req.tmin,
                "AWND": req.awnd,
            }
        ]
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model still loading")
    reference = state["reference"]
    frame = build_frame(req, reference)
    probability = float(state["model"].predict_proba(frame)[0, 1])
    base = reference["base_rate"]
    state["requests"] += 1

    quality = state.get("metrics", {}).get("model", {})
    return PredictResponse(
        delay_probability=round(probability, 4),
        base_rate=round(base, 4),
        relative_risk=round(probability / base, 2) if base else 0.0,
        band=_band(probability, base),
        route=f"{req.origin.upper()}-{req.dest.upper()}",
        model_quality={
            "roc_auc": round(quality.get("roc_auc", 0), 3),
            "top_decile_lift": round(quality.get("top_decile_lift", 0), 2),
            "calibration_error": round(quality.get("ece", 0), 4),
            "note": "Modest lift over a historical-rate baseline. Not travel advice.",
        },
    )


@app.get("/api/reference")
def reference() -> dict:
    """Airports, carriers and routes the UI populates its dropdowns from."""
    ref = state["reference"]
    return {"airports": ref["airports"], "carriers": ref["carriers"], "base_rate": ref["base_rate"]}


@app.get("/api/routes/{origin}")
def routes_from(origin: str) -> dict:
    """Destinations actually served from an origin, so the UI cannot offer a 404."""
    prefix = f"{origin.upper()}-"
    dests = sorted(k.split("-")[1] for k in state["reference"]["routes"] if k.startswith(prefix))
    return {"origin": origin.upper(), "destinations": dests}


@app.get("/health")
def health() -> dict:
    """Liveness and readiness in one probe: the service is only useful with a model."""
    ready = "model" in state
    return {
        "status": "ok" if ready else "loading",
        "model_loaded_ms": state.get("loaded_in_ms"),
        "requests_served": state.get("requests", 0),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
