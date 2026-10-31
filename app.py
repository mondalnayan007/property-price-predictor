from pathlib import Path
import json
from typing import Optional

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dhaka_apartment_price_model.cbm"
LOCATIONS_PATH = BASE_DIR / "locations.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model file missing: dhaka_apartment_price_model.cbm")
if not LOCATIONS_PATH.exists():
    raise FileNotFoundError("locations.json missing")

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
LOCATIONS = json.loads(LOCATIONS_PATH.read_text(encoding="utf-8"))
LOCATION_MAP = {item["location"]: item for item in LOCATIONS}

app = FastAPI(title="BanglaProperties Price Predictor", version="1.0.0")

MEDIAN_LATITUDE = 23.76629
MEDIAN_LONGITUDE = 90.39280
MEDIAN_BEDS = 3.0
MEDIAN_BATHS = 3.0

class PredictionRequest(BaseModel):
    location: str = Field(min_length=2, max_length=200)
    region: str = Field(min_length=1, max_length=100)
    sub_region: str = Field(min_length=1, max_length=100)
    area_sqft: float = Field(gt=0, le=100000)
    bedrooms: int = Field(ge=0, le=30)
    bathrooms: int = Field(ge=0, le=30)

class PredictionResponse(BaseModel):
    estimated_price_bdt: float
    estimated_price_lakh_bdt: float
    currency: str = "BDT"
    property_type: str = "Apartment"
    disclaimer: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "known_locations": len(LOCATIONS)}

@app.get("/locations.json")
def locations():
    return LOCATIONS

@app.get("/", response_class=HTMLResponse)
def home():
    # The page and API are served from the same host, so no separate CORS setup is needed.
    return HTML

@app.post("/predict", response_model=PredictionResponse)
def predict_price(payload: PredictionRequest):
    # Prefer the known dataset values for location categories to reduce category spelling errors.
    selected = LOCATION_MAP.get(payload.location.strip())
    if not selected:
        raise HTTPException(status_code=422, detail="Please choose a location from the dropdown.")

    row = pd.DataFrame([{
        "Location": selected["location"],
        "No. Beds": payload.bedrooms if payload.bedrooms is not None else MEDIAN_BEDS,
        "No. Baths": payload.bathrooms if payload.bathrooms is not None else MEDIAN_BATHS,
        "Area": payload.area_sqft,
        "Latitude": MEDIAN_LATITUDE,
        "Longitude": MEDIAN_LONGITUDE,
        "Region": selected["region"],
        "Sub-region": selected["sub_region"],
    }])
    try:
        estimate = max(0.0, float(model.predict(row)[0]))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not predict from these inputs. Please check the values.") from exc

    return PredictionResponse(
        estimated_price_bdt=round(estimate, 2),
        estimated_price_lakh_bdt=round(estimate / 100000, 2),
        disclaimer=(
            "Approximate estimate based on historical property listing prices, not verified sale transactions. "
            "Actual market prices may differ. This model currently supports apartments in its Dhaka-area dataset."
        ),
    )

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0b6b54">
<title>Property Price Predictor | BanglaProperties</title>
<style>
:root{--green:#08775c;--deep:#064e3b;--ink:#182a25;--muted:#65756f;--line:#dce7e2;--bg:#f3f8f5;--white:#fff;--gold:#f4c96b}
*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg);line-height:1.5}
.topbar{background:var(--white);border-bottom:1px solid var(--line)}.nav{max-width:1100px;margin:auto;padding:18px 22px;display:flex;align-items:center;justify-content:space-between;gap:16px}.brand{font-weight:850;letter-spacing:-.04em;font-size:21px;color:var(--deep)}.brand span{color:var(--green)}.pill{font-size:12px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;background:#e6f5ee;color:var(--deep);padding:7px 11px;border-radius:999px}
main{max-width:1100px;margin:0 auto;padding:48px 22px 56px}.hero{max-width:700px;margin-bottom:28px}.eyebrow{color:var(--green);font-weight:800;font-size:12px;letter-spacing:.15em;text-transform:uppercase}.hero h1{font-size:clamp(32px,5vw,49px);line-height:1.08;letter-spacing:-.045em;margin:12px 0}.hero p{font-size:16px;color:var(--muted);max-width:650px;margin:0}.layout{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(290px,.9fr);gap:22px;align-items:start}.card{background:var(--white);border:1px solid var(--line);border-radius:22px;padding:28px;box-shadow:0 12px 35px rgba(12,57,41,.045)}.card h2{font-size:20px;margin:0 0 6px;letter-spacing:-.025em}.sub{font-size:14px;color:var(--muted);margin:0 0 24px}.field{margin-bottom:17px}.field label{display:block;font-size:13px;font-weight:750;margin-bottom:7px}.field input,.field select{width:100%;height:48px;border:1px solid #cedbd5;border-radius:11px;padding:0 13px;font:inherit;color:var(--ink);background:#fff;outline:none}.field input:focus,.field select:focus{border-color:var(--green);box-shadow:0 0 0 3px rgba(8,119,92,.12)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.hint{font-size:12px;color:var(--muted);margin-top:6px}.button{border:0;background:var(--green);color:#fff;width:100%;height:51px;border-radius:12px;font:inherit;font-weight:800;cursor:pointer;transition:transform .15s,background .15s}.button:hover{background:var(--deep);transform:translateY(-1px)}.button:disabled{opacity:.65;cursor:wait;transform:none}.privacy{font-size:12px;color:var(--muted);text-align:center;margin:12px 0 0}.side{background:linear-gradient(150deg,#075b45,#0b7b5d);color:#fff;position:relative;overflow:hidden}.side:after{content:"৳";position:absolute;right:-8px;top:-45px;font-size:220px;font-weight:900;color:rgba(255,255,255,.055);line-height:1}.side h2{font-size:22px;position:relative;z-index:1}.side p{color:#d8eee5;font-size:14px;position:relative;z-index:1}.feature{display:flex;gap:12px;margin:23px 0;position:relative;z-index:1}.icon{width:34px;height:34px;flex:0 0 34px;border-radius:10px;background:rgba(255,255,255,.13);display:grid;place-items:center;font-weight:900}.feature strong{display:block;font-size:14px}.feature span{display:block;color:#d8eee5;font-size:12px;margin-top:2px}.result{display:none;margin-top:20px;border:1px solid #b9e2cd;background:#f0fbf5;border-radius:16px;padding:20px}.result.show{display:block}.result-label{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--green)}.amount{font-size:clamp(27px,4vw,36px);font-weight:900;letter-spacing:-.04em;color:var(--deep);margin:5px 0}.lakh{font-size:14px;color:#315d4b}.disclaimer{font-size:12px;color:#66776f;margin-top:13px}.error{display:none;margin-top:15px;padding:12px;border-radius:10px;background:#fff0ee;color:#9f2e20;font-size:13px}.error.show{display:block}.footer{text-align:center;font-size:12px;color:var(--muted);padding:0 20px 28px}
@media(max-width:800px){main{padding-top:34px}.layout{grid-template-columns:1fr}.side{order:2}.card{padding:22px}.hero p{font-size:15px}}@media(max-width:420px){.grid2{grid-template-columns:1fr;gap:0}.nav{padding:15px 17px}main{padding-left:15px;padding-right:15px}}
</style></head>
<body><header class="topbar"><div class="nav"><div class="brand">Bangla<span>Properties</span></div><div class="pill">Smart property estimate</div></div></header>
<main><section class="hero"><div class="eyebrow">Property price predictor</div><h1>Estimate your property’s value.</h1><p>Enter your apartment details to get an indicative price estimate based on historical property listing data.</p></section>
<div class="layout"><section class="card"><h2>Apartment details</h2><p class="sub">Fill in the details below to calculate an estimate.</p>
<form id="predictForm"><div class="field"><label for="location">Location in Dhaka *</label><select id="location" name="location" required><option value="">Loading locations…</option></select><div class="hint">Choose the closest matching location from the list.</div></div>
<div class="grid2"><div class="field"><label for="region">Region</label><input id="region" name="region" readonly placeholder="Auto-filled from location"></div><div class="field"><label for="sub_region">Sub-region</label><input id="sub_region" name="sub_region" readonly placeholder="Auto-filled from location"></div></div>
<div class="field"><label for="area_sqft">Apartment area (sq ft) *</label><input id="area_sqft" name="area_sqft" type="number" min="100" max="100000" step="1" placeholder="e.g. 1200" required></div>
<div class="grid2"><div class="field"><label for="bedrooms">Bedrooms *</label><select id="bedrooms" name="bedrooms" required><option value="">Select</option><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option></select></div><div class="field"><label for="bathrooms">Bathrooms *</label><select id="bathrooms" name="bathrooms" required><option value="">Select</option><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option></select></div></div>
<button class="button" id="submitBtn" type="submit">Predict property price →</button><p class="privacy">No account required. This estimate is not a formal valuation.</p></form>
<div class="error" id="errorBox" role="alert"></div><section class="result" id="resultBox" aria-live="polite"><div class="result-label">Estimated asking price</div><div class="amount" id="priceBdt">৳—</div><div class="lakh" id="priceLakh"></div><p class="disclaimer" id="disclaimer"></p></section></section>
<aside class="card side"><h2>A helpful starting point for your property decision.</h2><p>Get a quick estimate before you start comparing apartments.</p><div class="feature"><div class="icon">✓</div><div><strong>Location-aware estimate</strong><span>Uses a known location from the training dataset.</span></div></div><div class="feature"><div class="icon">⌂</div><div><strong>Based on your apartment details</strong><span>Area, bedrooms and bathrooms are included.</span></div></div><div class="feature"><div class="icon">৳</div><div><strong>Bangladeshi Taka</strong><span>Price is shown in BDT and lakh.</span></div></div></aside></div></main>
<footer class="footer">© BanglaProperties · Estimates are indicative only and may differ from actual market prices.</footer>
<script>
const locationSelect=document.getElementById('location');
const regionInput=document.getElementById('region');
const subRegionInput=document.getElementById('sub_region');
const form=document.getElementById('predictForm');
const button=document.getElementById('submitBtn');
const errorBox=document.getElementById('errorBox');
const resultBox=document.getElementById('resultBox');
let locationRecords=[];
const money = n => new Intl.NumberFormat('en-BD',{style:'currency',currency:'BDT',maximumFractionDigits:0}).format(n);
fetch('/locations.json').then(r=>{if(!r.ok) throw new Error('Could not load locations. Refresh the page.');return r.json()}).then(data=>{
 locationRecords=data;
 locationSelect.innerHTML='<option value="">Choose a location</option>';
 data.forEach(item=>{const option=document.createElement('option');option.value=item.location;option.textContent=item.location;locationSelect.appendChild(option)});
}).catch(()=>{locationSelect.innerHTML='<option value="">Locations could not load — refresh</option>';showError('Could not load locations. Please refresh the page.');});
locationSelect.addEventListener('change',()=>{const item=locationRecords.find(x=>x.location===locationSelect.value);regionInput.value=item?item.region:'';subRegionInput.value=item?item.sub_region:'';resultBox.classList.remove('show');errorBox.classList.remove('show')});
function showError(message){errorBox.textContent=message;errorBox.classList.add('show')}
form.addEventListener('submit',async event=>{
 event.preventDefault();errorBox.classList.remove('show');resultBox.classList.remove('show');
 const payload={location:locationSelect.value,region:regionInput.value,sub_region:subRegionInput.value,area_sqft:Number(document.getElementById('area_sqft').value),bedrooms:Number(document.getElementById('bedrooms').value),bathrooms:Number(document.getElementById('bathrooms').value)};
 if(!payload.location){showError('Please choose a location.');return}
 button.disabled=true;button.textContent='Calculating estimate…';
 try{const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Prediction failed. Please try again.');
 document.getElementById('priceBdt').textContent=money(data.estimated_price_bdt);
 document.getElementById('priceLakh').textContent='Approximately ৳'+Number(data.estimated_price_lakh_bdt).toLocaleString('en-BD')+' lakh';
 document.getElementById('disclaimer').textContent=data.disclaimer;
 resultBox.classList.add('show');resultBox.scrollIntoView({behavior:'smooth',block:'nearest'});
 }catch(error){showError(error.message||'Something went wrong. Please try again.')}finally{button.disabled=false;button.textContent='Predict property price →'}
});
</script></body></html>'''
