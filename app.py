from pathlib import Path
import json
import difflib
import re

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException, Query
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

LOCATIONS = json.loads(
    LOCATIONS_PATH.read_text(encoding="utf-8")
)

LOCATION_MAP = {
    item["location"]: item for item in LOCATIONS
}

app = FastAPI(
    title="BDProperties Price Predictor",
    version="2.0.0"
)

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
    return {
        "status": "ok",
        "model_loaded": True,
        "known_locations": len(LOCATIONS)
    }


@app.get("/locations.json")
def locations():
    return LOCATIONS


def normalize_location(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.casefold()
    ).strip()


def find_location_matches(query: str, limit: int = 6):
    """
    Find closest available location names.
    This is text similarity, not geographic distance.
    """

    query_norm = normalize_location(query)

    if not query_norm:
        return []

    substring = [
        item for item in LOCATIONS
        if query_norm in normalize_location(item["location"])
        or normalize_location(item["location"]) in query_norm
    ]

    scored = []

    q_tokens = set(query_norm.split())

    for item in LOCATIONS:
        name_norm = normalize_location(item["location"])

        score = difflib.SequenceMatcher(
            None,
            query_norm,
            name_norm
        ).ratio()

        n_tokens = set(name_norm.split())

        if q_tokens and n_tokens:
            score = max(
                score,
                len(q_tokens & n_tokens) /
                len(q_tokens | n_tokens)
            )

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen = set()

    for item in substring:
        if item["location"] not in seen:
            results.append(item)
            seen.add(item["location"])

        if len(results) >= limit:
            return results

    for score, item in scored:
        if score < 0.22:
            continue

        if item["location"] not in seen:
            results.append(item)
            seen.add(item["location"])

        if len(results) >= limit:
            break

    return results


@app.get("/locations/search")
def search_locations(
    q: str = Query(default="", max_length=200)
):
    query = q.strip()

    exact = LOCATION_MAP.get(query) if query else None

    return {
        "exact": exact,
        "matches": find_location_matches(query),
        "match_type": (
            "exact" if exact
            else "closest_name_matches"
        ),
        "note": (
            "Suggestions are based on name similarity, "
            "not geographic distance."
        )
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/predict", response_model=PredictionResponse)
def predict_price(payload: PredictionRequest):

    selected = LOCATION_MAP.get(
        payload.location.strip()
    )

    if not selected:
        suggestions = find_location_matches(
            payload.location.strip(),
            limit=5
        )

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "That exact location is not in the supported dataset. "
                    "Please choose one of the closest available matches."
                ),
                "suggestions": [
                    item["location"]
                    for item in suggestions
                ]
            }
        )

    row = pd.DataFrame([{
        "Location": selected["location"],
        "No. Beds": payload.bedrooms,
        "No. Baths": payload.bathrooms,
        "Area": payload.area_sqft,
        "Latitude": MEDIAN_LATITUDE,
        "Longitude": MEDIAN_LONGITUDE,
        "Region": selected["region"],
        "Sub-region": selected["sub_region"],
    }])

    try:
        estimate = max(
            0.0,
            float(model.predict(row)[0])
        )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not predict from these inputs. "
                "Please check the values."
            )
        ) from exc

    return PredictionResponse(
        estimated_price_bdt=round(estimate, 2),
        estimated_price_lakh_bdt=round(
            estimate / 100000,
            2
        ),
        disclaimer=(
            "This is an indicative estimate based on historical "
            "property listing prices, not verified sale transactions. "
            "Actual market prices may differ. "
            "The model currently supports apartments in its "
            "Dhaka-area dataset."
        )
    )


HTML = r'''<!doctype html>
<html lang="en">

<head>

<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<meta name="theme-color" content="#102a43">

<meta name="description"
content="Estimate apartment prices with BDProperties.">

<title>BDProperties | Property Price Estimator</title>

<link rel="preconnect" href="https://fonts.googleapis.com">

<link rel="preconnect"
href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap"
rel="stylesheet">

<style>

:root {
    --navy: #102a43;
    --green: #087e68;
    --green2: #056653;
    --mint: #e8f5f1;
    --ink: #172b3a;
    --muted: #657889;
    --line: #dce5eb;
    --page: #f4f7f9;
    --shadow: 0 22px 60px rgba(16,42,67,.09);
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--page);
    color: var(--ink);
    font-family: "DM Sans", sans-serif;
    line-height: 1.55;
}

.topbar {
    background: #fff;
    border-bottom: 1px solid var(--line);
    position: sticky;
    top: 0;
    z-index: 5;
}

.nav {
    max-width: 1160px;
    margin: auto;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font: 800 21px Manrope, sans-serif;
    color: var(--navy);
    letter-spacing: -.7px;
}

.mark {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: var(--navy);
    color: white;
    display: grid;
    place-items: center;
    font-size: 21px;
}

.brand span:last-child {
    color: var(--green);
}

.tag {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    color: var(--green2);
    background: var(--mint);
    padding: 8px 12px;
    border-radius: 30px;
}

main {
    max-width: 1160px;
    margin: auto;
    padding: 50px 24px 60px;
}

.hero {
    text-align: center;
    max-width: 720px;
    margin: 0 auto 34px;
}

.eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--green);
}

h1 {
    font: 800 clamp(34px,5vw,52px)/1.12 Manrope,sans-serif;
    letter-spacing: -1.8px;
    color: var(--navy);
    margin: 13px 0;
}

.hero p {
    color: var(--muted);
    font-size: 15px;
    margin: 0;
}

.layout {
    display: grid;
    grid-template-columns: 1.15fr .85fr;
    gap: 23px;
    align-items: start;
}

.card {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 21px;
    box-shadow: var(--shadow);
}

.form-card {
    padding: 29px;
}

.heading {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 24px;
}

.heading-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: var(--mint);
    color: var(--green2);
    display: grid;
    place-items: center;
    font-size: 21px;
}

h2 {
    font: 700 20px Manrope,sans-serif;
    color: var(--navy);
    margin: 0;
}

.heading p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 12px;
}

.field {
    margin-bottom: 18px;
}

label {
    display: block;
    font-size: 13px;
    font-weight: 800;
    color: #243b53;
    margin-bottom: 7px;
}

.control {
    width: 100%;
    height: 48px;
    border: 1px solid #cfdbe3;
    border-radius: 10px;
    padding: 0 13px;
    background: #fff;
    color: var(--ink);
    font: inherit;
    outline: none;
}

.control:focus {
    border-color: var(--green);
    box-shadow: 0 0 0 4px #087e681c;
}

.control::placeholder {
    color: #9aa8b3;
}

.location-tools,
.grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 11px;
}

.hint {
    font-size: 11px;
    color: var(--muted);
    margin-top: 7px;
}

.readonly {
    background: #f7f9fa;
    color: #657889;
}

.status {
    display: none;
    margin-top: 9px;
    padding: 10px 12px;
    border-radius: 9px;
    font-size: 12px;
}

.status.show {
    display: block;
}

.status.warn {
    background: #fff7e8;
    color: #79551a;
    border: 1px solid #f0dfb8;
}

.status.good {
    background: #edf8f2;
    color: #176344;
    border: 1px solid #d1ebdc;
}

.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 8px;
}

.suggestion {
    border: 1px solid #c9e4d9;
    background: #f3fbf7;
    color: #12624e;
    padding: 7px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
}

.suggestion:hover {
    background: #e3f4ec;
}

.primary {
    width: 100%;
    height: 52px;
    border: 0;
    border-radius: 11px;
    background: var(--green);
    color: #fff;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 8px 18px #087e6833;
}

.primary:hover {
    background: var(--green2);
}

.primary:disabled {
    opacity: .65;
    cursor: wait;
}

.note {
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    margin: 12px 0 0;
}

.side {
    padding: 29px;
    background: var(--navy);
    color: white;
    overflow: hidden;
    position: relative;
}

.side:after {
    content: "BD";
    position: absolute;
    right: -12px;
    top: 0;
    font: 800 145px Manrope,sans-serif;
    color: #ffffff09;
}

.kicker {
    font-size: 11px;
    color: #9bd7c4;
    letter-spacing: .14em;
    font-weight: 800;
    text-transform: uppercase;
    position: relative;
}

.side h2 {
    color: white;
    font-size: 25px;
    line-height: 1.25;
    margin: 12px 0;
    position: relative;
}

.side-intro {
    font-size: 13px;
    color: #c7d5df;
    position: relative;
}

.benefit {
    display: flex;
    gap: 12px;
    margin: 21px 0;
    position: relative;
}

.benefit-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #ffffff16;
    display: grid;
    place-items: center;
    color: #a9e1cf;
    flex: 0 0 34px;
}

.benefit strong {
    font-size: 13px;
    display: block;
}

.benefit span {
    font-size: 11px;
    color: #bdcbd5;
    display: block;
    margin-top: 3px;
}

.side-note {
    margin-top: 26px;
    padding: 13px;
    border-radius: 10px;
    background: #ffffff12;
    color: #c7d5df;
    font-size: 11px;
    position: relative;
}

.result {
    display: none;
    margin-top: 19px;
    padding: 21px;
    border: 1px solid #cce8d9;
    background: #f1fbf5;
    border-radius: 14px;
}

.result.show {
    display: block;
}

.result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--green2);
    font-weight: 800;
}

.amount {
    font: 800 clamp(28px,4vw,38px) Manrope,sans-serif;
    color: var(--navy);
    letter-spacing: -1px;
    overflow-wrap: anywhere;
    margin-top: 6px;
}

.lakh {
    font-size: 13px;
    color: #47715f;
}

.disclaimer {
    font-size: 11px;
    color: #697b72;
    border-top: 1px solid #dcefe3;
    padding-top: 12px;
    margin: 14px 0 0;
}

.error {
    display: none;
    margin-top: 13px;
    padding: 11px;
    border-radius: 9px;
    background: #fff1f0;
    color: #a83232;
    font-size: 12px;
}

.error.show {
    display: block;
}

.footer {
    text-align: center;
    color: #84929d;
    font-size: 11px;
    padding: 0 20px 25px;
}

@media(max-width:820px) {
    .layout {
        grid-template-columns: 1fr;
    }

    .side {
        order: 2;
    }
}

@media(max-width:520px) {
    .nav {
        padding: 13px 15px;
    }

    .brand {
        font-size: 18px;
    }

    .tag {
        font-size: 9px;
        padding: 7px 9px;
    }

    main {
        padding: 32px 14px 42px;
    }

    .form-card,
    .side {
        padding: 20px;
    }

    .location-tools {
        grid-template-columns: 1fr;
    }

    .grid2 {
        gap: 9px;
    }

    .control {
        height: 46px;
    }
}

</style>
</head>

<body>

<header class="topbar">

<nav class="nav">

<div class="brand">
    <span class="mark">⌂</span>
    <span>BD<span>Properties</span></span>
</div>

<div class="tag">SMART PROPERTY ESTIMATOR</div>

</nav>

</header>


<main>

<section class="hero">

<div class="eyebrow">
    A smarter way to estimate
</div>

<h1>
    Know your property's<br>
    estimated value.
</h1>

<p>
    Share a few apartment details and get an indicative
    price estimate powered by our property data model.
</p>

</section>


<div class="layout">

<section class="card form-card">

<div class="heading">

<div class="heading-icon">⌂</div>

<div>
    <h2>Apartment details</h2>
    <p>Enter the details below to calculate your estimate.</p>
</div>

</div>


<form id="predictForm">

<div class="field">

<label for="locationSearch">
    Property location *
</label>

<div class="location-tools">

<input
    class="control"
    id="locationSearch"
    type="search"
    placeholder="Search an area, e.g. Dhanmondi"
    autocomplete="off"
    required
>

<select
    class="control"
    id="location"
    required
    aria-label="Select location"
>
    <option value="">Select from list…</option>
</select>

</div>

<div class="hint">
    Search by area name or select from the available locations.
</div>

<div
    class="status"
    id="locationStatus"
    role="status"
    aria-live="polite"
></div>

<div
    class="suggestions"
    id="suggestions"
></div>

</div>


<div class="grid2">

<div class="field">

<label for="region">Region</label>

<input
    class="control readonly"
    id="region"
    readonly
    placeholder="Auto-filled"
>

</div>


<div class="field">

<label for="sub_region">Sub-region</label>

<input
    class="control readonly"
    id="sub_region"
    readonly
    placeholder="Auto-filled"
>

</div>

</div>


<div class="field">

<label for="area_sqft">
    Apartment area (sq ft) *
</label>

<input
    class="control"
    id="area_sqft"
    type="number"
    min="100"
    max="100000"
    step="1"
    placeholder="e.g. 1200"
    required
>

</div>


<div class="grid2">

<div class="field">

<label for="bedrooms">Bedrooms *</label>

<select
    class="control"
    id="bedrooms"
    required
>

<option value="">Select</option>
<option>1</option>
<option>2</option>
<option selected>3</option>
<option>4</option>
<option>5</option>
<option>6</option>
<option>7</option>
<option>8</option>

</select>

</div>


<div class="field">

<label for="bathrooms">Bathrooms *</label>

<select
    class="control"
    id="bathrooms"
    required
>

<option value="">Select</option>
<option>1</option>
<option>2</option>
<option selected>3</option>
<option>4</option>
<option>5</option>
<option>6</option>
<option>7</option>
<option>8</option>

</select>

</div>

</div>


<button
    class="primary"
    id="submitBtn"
    type="submit"
>
    Calculate estimated price →
</button>

<p class="note">
    Free estimate · No account required
</p>

</form>


<div
    class="error"
    id="errorBox"
    role="alert"
></div>


<section
    class="result"
    id="resultBox"
    aria-live="polite"
>

<div class="result-label">
    Estimated apartment asking price
</div>

<div
    class="amount"
    id="priceBdt"
>
    ৳—
</div>

<div
    class="lakh"
    id="priceLakh"
></div>

<p
    class="disclaimer"
    id="disclaimer"
></p>

</section>

</section>


<aside class="card side">

<div class="kicker">
    BDProperties price insight
</div>

<h2>
    A confident first step in your property search.
</h2>

<p class="side-intro">
    Understand an indicative apartment price before comparing your options.
</p>


<div class="benefit">

<div class="benefit-icon">⌖</div>

<div>
    <strong>Location-aware estimate</strong>
    <span>
        Choose a location represented in the model dataset.
    </span>
</div>

</div>


<div class="benefit">

<div class="benefit-icon">▦</div>

<div>
    <strong>Built around your apartment</strong>
    <span>
        Area, bedrooms and bathrooms shape the estimate.
    </span>
</div>

</div>


<div class="benefit">

<div class="benefit-icon">৳</div>

<div>
    <strong>Prices in Bangladeshi Taka</strong>
    <span>
        See your estimate in BDT and lakh.
    </span>
</div>

</div>


<div class="side-note">
    This tool provides an indicative estimate from historical
    listing data. It is not a formal valuation or a guaranteed sale price.
</div>

</aside>

</div>

</main>


<footer class="footer">
    © <span id="year"></span> BDProperties.
    All rights reserved. · Indicative estimates only.
</footer>


<script>

const sel = document.getElementById('location');

const search = document.getElementById('locationSearch');

const region = document.getElementById('region');

const sub = document.getElementById('sub_region');

const form = document.getElementById('predictForm');

const btn = document.getElementById('submitBtn');

const err = document.getElementById('errorBox');

const result = document.getElementById('resultBox');

const statusBox = document.getElementById('locationStatus');

const suggestions = document.getElementById('suggestions');


let records = [];

let chosen = null;

let timer = null;


document.getElementById('year').textContent =
    new Date().getFullYear();


const money = n => new Intl.NumberFormat(
    'en-BD',
    {
        style: 'currency',
        currency: 'BDT',
        maximumFractionDigits: 0
    }
).format(n);


function status(msg, type = 'warn') {

    statusBox.textContent = msg;

    statusBox.className = 'status show ' + type;

}


function clearStatus() {

    statusBox.textContent = '';

    statusBox.className = 'status';

}


function choose(item) {

    if (!item) return;

    chosen = item;

    sel.value = item.location;

    search.value = item.location;

    region.value = item.region || '';

    sub.value = item.sub_region || '';

    suggestions.innerHTML = '';

    status('Selected: ' + item.location, 'good');

    err.classList.remove('show');

    result.classList.remove('show');

}


function showMatches(items, exact = false) {

    suggestions.innerHTML = '';

    if (!items.length) {

        status(
            'No matching location was found in the supported dataset. Try another area name.'
        );

        return;

    }

    if (!exact) {

        status(
            'Exact location not found. Choose the closest name match below. Suggestions are based on name similarity, not geographic distance.'
        );

    }


    items.slice(0, 6).forEach(item => {

        let b = document.createElement('button');

        b.type = 'button';

        b.className = 'suggestion';

        b.textContent = item.location;

        b.addEventListener(
            'click',
            () => choose(item)
        );

        suggestions.appendChild(b);

    });

}


// Load available locations

fetch('/locations.json')

.then(r => {

    if (!r.ok) throw Error();

    return r.json();

})

.then(data => {

    records = data;

    sel.innerHTML =
        '<option value="">Select from list…</option>';

    data.forEach(item => {

        let o = document.createElement('option');

        o.value = item.location;

        o.textContent = item.location;

        sel.appendChild(o);

    });

})

.catch(() => {

    status(
        'Could not load locations. Please refresh the page.'
    );

});


// Dropdown selection

sel.addEventListener('change', () => {

    let item = records.find(
        x => x.location === sel.value
    );

    if (item) {

        choose(item);

    } else {

        chosen = null;

        region.value = '';

        sub.value = '';

    }

});


// Location search

search.addEventListener('input', () => {

    let q = search.value.trim();

    chosen = null;

    sel.value = '';

    region.value = '';

    sub.value = '';

    suggestions.innerHTML = '';

    err.classList.remove('show');

    result.classList.remove('show');


    if (!q) {

        clearStatus();

        return;

    }


    // Show direct matches immediately

    let direct = records.filter(
        x => x.location.toLowerCase().includes(
            q.toLowerCase()
        )
    ).slice(0, 6);


    if (direct.length) {

        showMatches(direct, true);

    }


    clearTimeout(timer);


    timer = setTimeout(async () => {

        try {

            let r = await fetch(
                '/locations/search?q=' +
                encodeURIComponent(q)
            );

            if (!r.ok) throw Error();

            let data = await r.json();

            if (search.value.trim() !== q) return;


            if (data.exact) {

                choose(data.exact);

                return;

            }


            showMatches(
                data.matches || [],
                false
            );

        } catch (e) {

            if (!direct.length) {

                status(
                    'Search is temporarily unavailable. You can still choose from the dropdown.'
                );

            }

        }

    }, 220);

});


// Predict price

form.addEventListener('submit', async e => {

    e.preventDefault();

    err.classList.remove('show');

    result.classList.remove('show');


    if (!chosen || sel.value !== chosen.location) {

        err.textContent =
            'Please choose a location from the dropdown or suggested matches.';

        err.classList.add('show');

        return;

    }


    let payload = {

        location: chosen.location,

        region: chosen.region,

        sub_region: chosen.sub_region,

        area_sqft: Number(
            document.getElementById('area_sqft').value
        ),

        bedrooms: Number(
            document.getElementById('bedrooms').value
        ),

        bathrooms: Number(
            document.getElementById('bathrooms').value
        )

    };


    btn.disabled = true;

    btn.textContent = 'Calculating estimate…';


    try {

        let r = await fetch('/predict', {

            method: 'POST',

            headers: {
                'Content-Type': 'application/json'
            },

            body: JSON.stringify(payload)

        });


        let data = await r.json();


        if (!r.ok) {

            throw Error(
                typeof data.detail === 'string'
                    ? data.detail
                    : (
                        data.detail?.message ||
                        'Prediction failed. Please try again.'
                    )
            );

        }


        document.getElementById('priceBdt').textContent =
            money(data.estimated_price_bdt);


        document.getElementById('priceLakh').textContent =
            'Approximately ৳' +
            Number(
                data.estimated_price_lakh_bdt
            ).toLocaleString('en-BD') +
            ' lakh';


        document.getElementById('disclaimer').textContent =
            data.disclaimer;


        result.classList.add('show');

        result.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest'
        });


    } catch (ex) {

        err.textContent =
            ex.message ||
            'Something went wrong. Please try again.';

        err.classList.add('show');


    } finally {

        btn.disabled = false;

        btn.textContent =
            'Calculate estimated price →';

    }

});

</script>

</body>
</html>'''