from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "property_price_model.cbm"
METADATA_PATH = BASE_DIR / "model_metadata.json"

app = FastAPI(title="Property Price Predictor", version="2.0")
model = CatBoostRegressor()
metadata = {}
model_ready = False

if MODEL_PATH.exists() and METADATA_PATH.exists():
    model.load_model(str(MODEL_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model_ready = True

FEATURES = [
    "Type", "Location", "No. Beds", "No. Baths", "Area",
    "Latitude", "Longitude", "Region", "Sub-region",
]

def money_bdt(value: float) -> str:
    return "৳" + f"{round(value):,}"

def location_options(selected: str = "") -> str:
    rows = metadata.get("locations", [])
    out = []
    for row in rows:
        name = str(row.get("Location") or "")
        safe_name = (name.replace("&", "&amp;").replace('"', "&quot;")
                     .replace("<", "&lt;").replace(">", "&gt;"))
        sel = " selected" if name == selected else ""
        out.append(f'<option value="{safe_name}"{sel}>{safe_name}</option>')
    return "\n".join(out)

def type_options(selected: str = "") -> str:
    values = metadata.get("types", ["Apartment"])
    out = []
    for value in values:
        safe = str(value).replace("&", "&amp;").replace('"', "&quot;")
        sel = " selected" if value == selected else ""
        out.append(f'<option value="{safe}"{sel}>{safe}</option>')
    return "\n".join(out)

def page(error: str = "", result: Optional[float] = None,
         form: Optional[dict] = None) -> str:
    form = form or {}
    if not model_ready:
        error = "Model files missing. Run train_model.py first and deploy the generated model files."
    error_html = f'<div class="alert error">{error}</div>' if error else ""
    result_html = ""
    if result is not None:
        result_html = f"""
        <section class="result" aria-live="polite">
          <div class="result-label">Estimated listing price</div>
          <div class="price">{money_bdt(result)}</div>
          <div class="subprice">Approximately ৳{result/100000:.2f} lakh</div>
          <p>This is a model estimate, not a confirmed market or transaction price.</p>
        </section>"""
    location = str(form.get("location", ""))
    prop_type = str(form.get("property_type", "Apartment"))
    beds = form.get("bedrooms", "3")
    baths = form.get("bathrooms", "2")
    area = form.get("area_sqft", "1200")
    disabled = "" if model_ready else " disabled"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Property Price Predictor | Bangladesh</title>
<style>
:root{{--ink:#14213d;--muted:#667085;--line:#e5eaf2;--blue:#2357d9;--pale:#f4f7ff;}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#f3f6ff,#f8fbff 55%,#edf8f5);font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);min-height:100vh}}
header{{padding:24px 6%;display:flex;justify-content:space-between;align-items:center;gap:12px}} .brand{{font-weight:850;font-size:20px;letter-spacing:-.5px}} .pill{{font-size:12px;font-weight:700;color:#176b4d;background:#e0f7ed;padding:8px 12px;border-radius:30px}}
main{{max-width:1000px;margin:24px auto 64px;padding:0 20px}} .hero{{text-align:center;margin-bottom:28px}} .eyebrow{{color:var(--blue);font-weight:800;font-size:12px;letter-spacing:2px;text-transform:uppercase}} h1{{font-size:clamp(30px,5vw,48px);line-height:1.08;letter-spacing:-1.5px;margin:12px 0}} .hero p{{color:var(--muted);font-size:16px;line-height:1.6;max-width:650px;margin:0 auto}}
.layout{{display:grid;grid-template-columns:1.25fr .75fr;gap:22px;align-items:start}} .card{{background:#fff;border:1px solid var(--line);border-radius:22px;padding:26px;box-shadow:0 18px 50px #1d3b7210}} .card h2{{font-size:20px;margin:0 0 6px}} .hint{{font-size:13px;color:var(--muted);margin:0 0 22px}} .fields{{display:grid;grid-template-columns:1fr 1fr;gap:17px}} .field.full{{grid-column:1/-1}} label{{display:block;font-size:13px;font-weight:750;margin-bottom:8px}} input,select{{width:100%;border:1px solid #d8dfeb;border-radius:11px;padding:13px 12px;font:inherit;background:#fff;color:var(--ink);outline:none}} input:focus,select:focus{{border-color:var(--blue);box-shadow:0 0 0 3px #2357d91a}} .btn{{width:100%;border:0;border-radius:12px;background:var(--blue);color:white;font-size:15px;font-weight:800;padding:15px;margin-top:22px;cursor:pointer;box-shadow:0 8px 20px #2357d930}} .btn:hover{{background:#1746bd}} .btn:disabled{{opacity:.5;cursor:not-allowed}} .side{{background:var(--ink);color:white;border-radius:22px;padding:25px}} .side h3{{margin:0 0 12px;font-size:19px}} .side p,.side li{{color:#d4dcec;font-size:14px;line-height:1.65}} .side ul{{padding-left:19px}} .note{{font-size:12px;color:#aebbd0;border-top:1px solid #ffffff22;padding-top:16px;margin-top:20px}} .alert{{padding:12px 14px;border-radius:12px;margin:0 0 18px;font-size:14px}} .error{{background:#fff0f0;color:#9e2424;border:1px solid #ffd0d0}} .result{{margin-top:20px;border:1px solid #b9e8d3;background:#f0fff7;border-radius:18px;padding:22px}} .result-label{{font-size:13px;font-weight:800;color:#21724f}} .price{{font-size:clamp(27px,4vw,38px);font-weight:900;letter-spacing:-1px;margin:8px 0;color:#145c3c}} .subprice{{font-weight:750;color:#287854}} .result p{{font-size:12px;color:#557466;line-height:1.5}} footer{{text-align:center;color:#7b879a;font-size:12px;padding:0 20px 26px}}
@media(max-width:760px){{.layout{{grid-template-columns:1fr}}header{{padding:18px 20px}}main{{margin-top:12px}}.card{{padding:20px}}}} @media(max-width:440px){{.fields{{grid-template-columns:1fr}}.field.full{{grid-column:auto}}}}
</style></head><body>
<header><div class="brand">⌂ Property<span style="color:var(--blue)">Price</span></div><span class="pill">ML POWERED ESTIMATE</span></header>
<main><section class="hero"><div class="eyebrow">Bangladesh property estimator</div><h1>Estimate your property's price.</h1><p>Enter the property details below to get an estimated listing price based on the available property dataset.</p></section>
{error_html}
<div class="layout"><section class="card"><h2>Property details</h2><p class="hint">Fields marked with * are required.</p>
<form method="post" action="/predict"><div class="fields">
<div class="field full"><label for="property_type">Property type *</label><select id="property_type" name="property_type" required>{type_options(prop_type)}</select></div>
<div class="field full"><label for="location">Location *</label><select id="location" name="location" required><option value="">Select a location</option>{location_options(location)}</select></div>
<div class="field full"><label for="area_sqft">Area (square feet) *</label><input id="area_sqft" name="area_sqft" type="number" min="100" max="1000000" step="1" value="{area}" required></div>
<div class="field"><label for="bedrooms">Bedrooms</label><input id="bedrooms" name="bedrooms" type="number" min="0" max="30" step="1" value="{beds}"></div>
<div class="field"><label for="bathrooms">Bathrooms</label><input id="bathrooms" name="bathrooms" type="number" min="0" max="30" step="1" value="{baths}"></div>
</div><button class="btn" type="submit"{disabled}>Predict property price →</button></form>
{result_html}</section>
<aside class="side"><h3>How it works</h3><p>The model learns patterns from property listings, including location, property type, size and room counts.</p><ul><li>Choose a location from the dataset.</li><li>Enter the property area and room details.</li><li>Get an estimated price in Bangladeshi taka.</li></ul><div class="note">Estimates are informational only. Actual prices depend on building condition, floor, amenities, ownership documents and market conditions. The dataset contains asking/listing prices, not verified sale transactions.</div></aside></div></main><footer>Property Price Predictor · Estimates are not a valuation or financial advice.</footer>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return page()

@app.get("/health")
def health():
    return {"status": "ok" if model_ready else "model_missing", "model_loaded": model_ready}

@app.post("/predict", response_class=HTMLResponse)
def predict(
    property_type: str = Form(...),
    location: str = Form(...),
    area_sqft: float = Form(...),
    bedrooms: Optional[float] = Form(None),
    bathrooms: Optional[float] = Form(None),
):
    if not model_ready:
        return HTMLResponse(page("The model has not been trained/deployed yet."), status_code=503)
    if area_sqft <= 0 or area_sqft > 1_000_000:
        return HTMLResponse(page("Please enter a valid area in square feet.", form={
            "property_type": property_type, "location": location,
            "area_sqft": area_sqft, "bedrooms": bedrooms or "", "bathrooms": bathrooms or ""
        }), status_code=400)

    match = next((r for r in metadata.get("locations", []) if r.get("Location") == location), None)
    if not match:
        return HTMLResponse(page("Please select a location from the list."), status_code=400)

    row = {
        "Type": property_type,
        "Location": location,
        "No. Beds": float(bedrooms) if bedrooms not in (None, "") else np.nan,
        "No. Baths": float(bathrooms) if bathrooms not in (None, "") else np.nan,
        "Area": float(area_sqft),
        "Latitude": match.get("Latitude") if match.get("Latitude") is not None else np.nan,
        "Longitude": match.get("Longitude") if match.get("Longitude") is not None else np.nan,
        "Region": str(match.get("Region") or "Unknown"),
        "Sub-region": str(match.get("Sub-region") or "Unknown"),
    }
    frame = pd.DataFrame([row], columns=FEATURES)
    try:
        predicted = float(np.expm1(model.predict(frame)[0]))
        predicted = max(0.0, predicted)
    except Exception:
        return HTMLResponse(page("Prediction failed. Please check the selected details and try again."), status_code=500)
    return HTMLResponse(page(result=predicted, form={
        "property_type": property_type, "location": location,
        "area_sqft": area_sqft, "bedrooms": bedrooms if bedrooms is not None else "",
        "bathrooms": bathrooms if bathrooms is not None else "",
    }))

@app.post("/api/predict")
def api_predict(payload: dict):
    """Optional JSON endpoint for later integrations."""
    if not model_ready:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    try:
        location = str(payload["location"])
        match = next((r for r in metadata.get("locations", []) if r.get("Location") == location), None)
        if not match:
            return JSONResponse({"error": "Unknown location"}, status_code=400)
        row = {
            "Type": str(payload.get("type", "Apartment")),
            "Location": location,
            "No. Beds": float(payload["bedrooms"]) if payload.get("bedrooms") not in (None, "") else np.nan,
            "No. Baths": float(payload["bathrooms"]) if payload.get("bathrooms") not in (None, "") else np.nan,
            "Area": float(payload["area_sqft"]),
            "Latitude": match.get("Latitude") if match.get("Latitude") is not None else np.nan,
            "Longitude": match.get("Longitude") if match.get("Longitude") is not None else np.nan,
            "Region": str(match.get("Region") or "Unknown"),
            "Sub-region": str(match.get("Sub-region") or "Unknown"),
        }
        predicted = max(0.0, float(np.expm1(model.predict(pd.DataFrame([row], columns=FEATURES))[0])))
        return {"estimated_price_bdt": round(predicted), "estimated_price_lakh": round(predicted / 100000, 2),
                "note": "Estimated listing price, not a verified transaction price."}
    except (KeyError, ValueError, TypeError) as exc:
        return JSONResponse({"error": f"Invalid input: {exc}"}, status_code=400)
