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
    raise FileNotFoundError(
        "Model file missing: dhaka_apartment_price_model.cbm"
    )

if not LOCATIONS_PATH.exists():
    raise FileNotFoundError("locations.json missing")

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

LOCATIONS = json.loads(
    LOCATIONS_PATH.read_text(encoding="utf-8")
)

LOCATION_MAP = {
    item["location"]: item
    for item in LOCATIONS
}

app = FastAPI(
    title="BDProperties Price Predictor",
    version="2.1.0"
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
    """Find similar location names, not geographic distances."""

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
                len(q_tokens & n_tokens)
                / len(q_tokens | n_tokens)
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
            "exact" if exact else "closest_name_matches"
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
        raise HTTPException(
            status_code=422,
            detail="Please choose a location from the dropdown."
        )

    row = pd.DataFrame([{
        "Location": selected["location"],
        "No. Beds": (
            payload.bedrooms
            if payload.bedrooms is not None
            else MEDIAN_BEDS
        ),
        "No. Baths": (
            payload.bathrooms
            if payload.bathrooms is not None
            else MEDIAN_BATHS
        ),
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
            "Approximate estimate based on historical property "
            "listing prices, not verified sale transactions. "
            "Actual market prices may differ. This model "
            "currently supports apartments in its Dhaka-area dataset."
        ),
    )


HTML = r'''<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<meta name="theme-color" content="#185F35">

<meta name="description"
content="Estimate apartment prices with BDProperties.">

<title>BDProperties | Property Price Estimator</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap"
rel="stylesheet">

<style>

:root {
    --green: #185F35;
    --green-dark: #104725;
    --green-light: #E8F4EB;
    --navy: #123B27;
    --ink: #172B3A;
    --muted: #657889;
    --line: #DCE5EB;
    --page: #F4F7F5;
    --white: #FFFFFF;
    --shadow: 0 22px 60px rgba(24,95,53,.10);
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background: var(--page);
    color: var(--ink);
    font-family: "DM Sans", sans-serif;
    line-height: 1.55;
}

button,
input,
select {
    font: inherit;
}

.topbar {
    background: #FFFFFF;
    border-bottom: 1px solid var(--line);
    position: sticky;
    top: 0;
    z-index: 10;
}

.nav {
    max-width: 1160px;
    margin: auto;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font: 800 21px "Manrope", sans-serif;
    color: var(--navy);
    letter-spacing: -.7px;
}

.mark {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: var(--green);
    color: white;
    display: grid;
    place-items: center;
    font-size: 25px;
    animation: floatIcon 4s ease-in-out infinite;
}

.brand-name {
    color: var(--green);
}

.tag {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    color: var(--green);
    background: var(--green-light);
    padding: 8px 12px;
    border-radius: 30px;
}

main {
    max-width: 1160px;
    margin: auto;
    padding: 48px 24px 60px;
}

.hero {
    text-align: center;
    max-width: 850px;
    margin: 0 auto 36px;
    animation: fadeUp .7s ease both;
}

.eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--green);
}

h1 {
    font: 800 clamp(34px, 5vw, 54px)/1.12
        "Manrope", sans-serif;
    letter-spacing: -1.8px;
    color: var(--navy);
    margin: 14px 0;
}

.hero-description {
    color: var(--muted);
    font-size: 15px;
    max-width: 650px;
    margin: 0 auto;
}

/* PROPERTY IMAGE */

.hero-photo {
    position: relative;
    margin: 28px auto 0;
    width: 100%;
    height: 250px;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 18px 40px rgba(18,59,39,.17);
    animation: fadeUp .9s ease both;
}

.hero-photo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform .8s ease;
}

.hero-photo:hover img {
    transform: scale(1.04);
}

.photo-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        90deg,
        rgba(8,35,19,.55),
        transparent 85%
    );
    pointer-events: none;
}

.photo-caption {
    position: absolute;
    left: 18px;
    bottom: 18px;
    background: rgba(13,43,27,.88);
    color: white;
    border: 1px solid #FFFFFF38;
    padding: 11px 15px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 9px;
}

.photo-dot {
    width: 9px;
    height: 9px;
    background: #8BD5A2;
    border-radius: 50%;
    box-shadow: 0 0 0 4px #8BD5A226;
}

/* MAIN LAYOUT */

.layout {
    display: grid;
    grid-template-columns: 1.15fr .85fr;
    gap: 23px;
    align-items: start;
}

.card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 21px;
    box-shadow: var(--shadow);
    animation: fadeUp .75s ease both;
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
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--green-light);
    color: var(--green);
    display: grid;
    place-items: center;
    font-size: 23px;
}

h2 {
    font: 700 20px "Manrope", sans-serif;
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
    color: #243B53;
    margin-bottom: 7px;
}

.control {
    width: 100%;
    height: 48px;
    border: 1px solid #CFDBE3;
    border-radius: 10px;
    padding: 0 13px;
    background: #FFFFFF;
    color: var(--ink);
    outline: none;
    transition: border-color .2s, box-shadow .2s;
}

.control:focus {
    border-color: var(--green);
    box-shadow: 0 0 0 4px #185F351C;
}

.control::placeholder {
    color: #9AA8B3;
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
    background: #F7F9FA;
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
    background: #FFF7E8;
    color: #79551A;
    border: 1px solid #F0DFB8;
}

.status.good {
    background: #EDF8F2;
    color: #176344;
    border: 1px solid #D1EBDC;
}

.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 8px;
}

.suggestion {
    border: 1px solid #C9E4D9;
    background: #F3FBF7;
    color: var(--green);
    padding: 7px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: background .2s, transform .2s;
}

.suggestion:hover {
    background: #E3F4EC;
    transform: translateY(-2px);
}

/* GREEN BUTTON */

.primary {
    width: 100%;
    min-height: 52px;
    border: 0;
    border-radius: 11px;
    background: #185F35;
    color: #FFFFFF;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 8px 18px #185F3533;
    transition:
        background .2s ease,
        transform .2s ease,
        box-shadow .2s ease;
}

.primary:hover {
    background: #104725;
    transform: translateY(-2px);
    box-shadow: 0 12px 24px #185F3540;
}

.primary:active {
    transform: translateY(0);
}

.primary:disabled {
    opacity: .65;
    cursor: wait;
    transform: none;
}

.note {
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    margin: 12px 0 0;
}

/* RIGHT INFORMATION PANEL */

.side {
    padding: 29px;
    background: var(--navy);
    color: white;
    overflow: hidden;
    position: relative;
    animation-delay: .12s;
}

.side:after {
    content: "BD";
    position: absolute;
    right: -12px;
    top: 0;
    font: 800 145px "Manrope", sans-serif;
    color: #FFFFFF09;
    pointer-events: none;
}

.kicker {
    font-size: 11px;
    color: #9BD7C4;
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
    color: #C7D5DF;
    position: relative;
}

.benefit {
    display: flex;
    gap: 12px;
    margin: 21px 0;
    position: relative;
}

.benefit-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: #FFFFFF16;
    display: grid;
    place-items: center;
    color: #A9E1CF;
    flex: 0 0 36px;
}

.benefit strong {
    font-size: 13px;
    display: block;
}

.benefit span {
    font-size: 11px;
    color: #BDCBD5;
    display: block;
    margin-top: 3px;
}

.side-note {
    margin-top: 26px;
    padding: 13px;
    border-radius: 10px;
    background: #FFFFFF12;
    color: #C7D5DF;
    font-size: 11px;
    position: relative;
}

/* PRICE RESULT */

.result {
    display: none;
    margin-top: 19px;
    padding: 21px;
    border: 1px solid #CCE8D9;
    background: #F1FBF5;
    border-radius: 14px;
}

.result.show {
    display: block;
    animation: fadeUp .35s ease both;
}

.result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--green);
    font-weight: 800;
}

.amount {
    font: 800 clamp(28px, 4vw, 38px) "Manrope", sans-serif;
    color: var(--navy);
    letter-spacing: -1px;
    overflow-wrap: anywhere;
    margin-top: 6px;
}

.lakh {
    font-size: 13px;
    color: #47715F;
}

.disclaimer {
    font-size: 11px;
    color: #697B72;
    border-top: 1px solid #DCEFE3;
    padding-top: 12px;
    margin: 14px 0 0;
}

.error {
    display: none;
    margin-top: 13px;
    padding: 11px;
    border-radius: 9px;
    background: #FFF1F0;
    color: #A83232;
    font-size: 12px;
}

.error.show {
    display: block;
}

.footer {
    text-align: center;
    color: #84929D;
    font-size: 11px;
    padding: 0 20px 25px;
}

/* ANIMATIONS */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes floatIcon {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-5px);
    }
}

/* RESPONSIVE DESIGN */

@media (max-width: 820px) {
    .layout {
        grid-template-columns: 1fr;
    }

    .side {
        order: 2;
    }
}

@media (max-width: 520px) {
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

    .hero-photo {
        height: 165px;
        border-radius: 14px;
    }

    .photo-caption {
        left: 9px;
        bottom: 9px;
        font-size: 10px;
        padding: 8px 10px;
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

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}

</style>
</head>

<body>

<header class="topbar">
    <nav class="nav">
        <div class="brand">
            <span class="mark">⌂</span>
            <span>BD<span class="brand-name">Properties</span></span>
        </div>

        <div class="tag">SMART PROPERTY ESTIMATOR</div>
    </nav>
</header>

<main>

    <section class="hero">

        <div class="eyebrow">A smarter way to estimate</div>

        <h1>
            Know your property's<br>
            estimated value.
        </h1>

        <p class="hero-description">
            Share a few apartment details and get an indicative
            price estimate powered by our property data model.
        </p>

        <div class="hero-photo">
            <img
                src="https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=1400&q=85"
                alt="Modern apartment interior"
                loading="eager"
            >

            <div class="photo-overlay"></div>

            <div class="photo-caption">
                <span class="photo-dot"></span>
                Find your next place with confidence
            </div>
        </div>

    </section>

    <div class="layout">

        <!-- PREDICTION FORM -->

        <section class="card form-card">

            <div class="heading">

                <div class="heading-icon">⌂</div>

                <div>
                    <h2>Apartment details</h2>
                    <p>
                        Enter the details below to calculate
                        your estimate.
                    </p>
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
                            <option value="">
                                Select from list…
                            </option>
                        </select>

                    </div>

                    <div class="hint">
                        Search by area name or select from
                        the available locations.
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

                        <label for="bedrooms">
                            Bedrooms *
                        </label>

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

                        <label for="bathrooms">
                            Bathrooms *
                        </label>

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

                <div class="amount" id="priceBdt">
                    ৳—
                </div>

                <div class="lakh" id="priceLakh"></div>

                <p
                    class="disclaimer"
                    id="disclaimer"
                ></p>

            </section>

        </section>

        <!-- INFORMATION PANEL -->

        <aside class="card side">

            <div class="kicker">
                BDProperties price insight
            </div>

            <h2>
                A confident first step in your property search.
            </h2>

            <p class="side-intro">
                Understand an indicative apartment price
                before comparing your options.
            </p>

            <div class="benefit">

                <div class="benefit-icon">⌖</div>

                <div>
                    <strong>Location-aware estimate</strong>
                    <span>
                        Choose a location represented
                        in the model dataset.
                    </span>
                </div>

            </div>

            <div class="benefit">

                <div class="benefit-icon">▦</div>

                <div>
                    <strong>Built around your apartment</strong>
                    <span>
                        Area, bedrooms and bathrooms
                        shape the estimate.
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
                This tool provides an indicative estimate
                from historical listing data. It is not a
                formal valuation or a guaranteed sale price.
            </div>

        </aside>

    </div>

</main>

<footer class="footer">
    © <span id="year"></span> BDProperties.
    All rights reserved. · Indicative estimates only.
</footer>

<script>

const sel = document.getElementById("location");
const search = document.getElementById("locationSearch");
const region = document.getElementById("region");
const sub = document.getElementById("sub_region");

const form = document.getElementById("predictForm");
const btn = document.getElementById("submitBtn");
const err = document.getElementById("errorBox");
const result = document.getElementById("resultBox");

const statusBox = document.getElementById("locationStatus");
const suggestions = document.getElementById("suggestions");

let records = [];
let chosen = null;
let timer = null;

document.getElementById("year").textContent =
    new Date().getFullYear();


const money = n =>
    new Intl.NumberFormat("en-BD", {
        style: "currency",
        currency: "BDT",
        maximumFractionDigits: 0
    }).format(n);


function status(msg, type = "warn") {
    statusBox.textContent = msg;
    statusBox.className = "status show " + type;
}


function clearStatus() {
    statusBox.textContent = "";
    statusBox.className = "status";
}


function choose(item) {

    if (!item) return;

    chosen = item;

    sel.value = item.location;
    search.value = item.location;

    region.value = item.region || "";
    sub.value = item.sub_region || "";

    suggestions.innerHTML = "";

    status(
        "Selected: " + item.location,
        "good"
    );

    err.classList.remove("show");
    result.classList.remove("show");
}


function showMatches(items, exact = false) {

    suggestions.innerHTML = "";

    if (!items.length) {
        status(
            "No matching location was found in the supported dataset. Try another area name."
        );
        return;
    }

    if (!exact) {
        status(
            "Exact location not found. Choose the closest name match below. Suggestions are based on name similarity, not geographic distance."
        );
    }

    items.slice(0, 6).forEach(item => {

        const button = document.createElement("button");

        button.type = "button";
        button.className = "suggestion";
        button.textContent = item.location;

        button.addEventListener("click", () => {
            choose(item);
        });

        suggestions.appendChild(button);

    });
}


/* LOAD AVAILABLE LOCATIONS */

fetch("/locations.json")

    .then(response => {

        if (!response.ok) {
            throw new Error("Could not load locations");
        }

        return response.json();

    })

    .then(data => {

        records = data;

        sel.innerHTML =
            '<option value="">Select from list…</option>';

        data.forEach(item => {

            const option = document.createElement("option");

            option.value = item.location;
            option.textContent = item.location;

            sel.appendChild(option);

        });

    })

    .catch(() => {

        status(
            "Could not load locations. Please refresh the page."
        );

    });


/* DROPDOWN LOCATION SELECTION */

sel.addEventListener("change", () => {

    const item = records.find(
        x => x.location === sel.value
    );

    if (item) {
        choose(item);
    } else {
        chosen = null;
        region.value = "";
        sub.value = "";
    }

});


/* SEARCH LOCATIONS */

search.addEventListener("input", () => {

    const q = search.value.trim();

    chosen = null;

    sel.value = "";
    region.value = "";
    sub.value = "";

    suggestions.innerHTML = "";

    err.classList.remove("show");
    result.classList.remove("show");

    if (!q) {
        clearStatus();
        return;
    }

    const direct = records
        .filter(item =>
            item.location.toLowerCase()
                .includes(q.toLowerCase())
        )
        .slice(0, 6);

    if (direct.length) {
        showMatches(direct, true);
    }

    clearTimeout(timer);

    timer = setTimeout(async () => {

        try {

            const response = await fetch(
                "/locations/search?q=" +
                encodeURIComponent(q)
            );

            if (!response.ok) {
                throw new Error("Search failed");
            }

            const data = await response.json();

            if (search.value.trim() !== q) {
                return;
            }

            if (data.exact) {
                choose(data.exact);
                return;
            }

            showMatches(data.matches || [], false);

        } catch (error) {

            if (!direct.length) {
                status(
                    "Search is temporarily unavailable. You can still choose from the dropdown."
                );
            }

        }

    }, 220);

});


/* PRICE PREDICTION FORM */

form.addEventListener("submit", async event => {

    event.preventDefault();

    err.classList.remove("show");
    result.classList.remove("show");

    if (!chosen || sel.value !== chosen.location) {

        err.textContent =
            "Please choose a location from the dropdown or suggested matches.";

        err.classList.add("show");

        return;
    }

    const area = Number(
        document.getElementById("area_sqft").value
    );

    const bedrooms = Number(
        document.getElementById("bedrooms").value
    );

    const bathrooms = Number(
        document.getElementById("bathrooms").value
    );

    if (
        !Number.isFinite(area) ||
        area < 100 ||
        area > 100000
    ) {

        err.textContent =
            "Please enter an apartment area between 100 and 100000 square feet.";

        err.classList.add("show");

        return;
    }

    const payload = {
        location: chosen.location,
        region: chosen.region,
        sub_region: chosen.sub_region,
        area_sqft: area,
        bedrooms: bedrooms,
        bathrooms: bathrooms
    };

    btn.disabled = true;
    btn.textContent = "Calculating estimate…";

    try {

        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                typeof data.detail === "string"
                    ? data.detail
                    : (
                        data.detail?.message ||
                        "Prediction failed. Please try again."
                    )
            );

        }

        document.getElementById("priceBdt").textContent =
            money(data.estimated_price_bdt);

        document.getElementById("priceLakh").textContent =
            "Approximately ৳" +
            Number(
                data.estimated_price_lakh_bdt
            ).toLocaleString("en-BD") +
            " lakh";

        document.getElementById("disclaimer").textContent =
            data.disclaimer;

        result.classList.add("show");

        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    } catch (error) {

        err.textContent =
            error.message ||
            "Something went wrong. Please try again.";

        err.classList.add("show");

    } finally {

        btn.disabled = false;

        btn.textContent =
            "Calculate estimated price →";

    }

});

</script>

</body>
</html>'''