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

<title>BDProperties — Find Your Place</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
:root {
    --green: #185F35;
    --green-dark: #104725;
    --green-light: #EAF3EC;
    --cream: #F8F8F3;
    --white: #FFFFFF;
    --ink: #1C2922;
    --muted: #737D76;
    --border: #E5E9E3;
    --gold: #C6A66B;
    --shadow: 0 18px 50px rgba(24, 55, 35, .08);
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    font-family: "DM Sans", sans-serif;
    color: var(--ink);
    background: var(--cream);
    line-height: 1.6;
}

button, input, select {
    font: inherit;
}

button {
    -webkit-tap-highlight-color: transparent;
}

.container {
    width: min(1160px, calc(100% - 48px));
    margin: 0 auto;
}

/* NAVIGATION */

.navbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(255,255,255,.95);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(24,95,53,.08);
}

.nav-inner {
    min-height: 78px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 11px;
    text-decoration: none;
    color: var(--green);
}

.logo-icon {
    height: 43px;
    width: 43px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    color: white;
    background: var(--green);
    font-size: 27px;
    font-weight: 700;
    box-shadow: 0 6px 15px rgba(24,95,53,.19);
}

.logo-text {
    font: 800 21px "Manrope", sans-serif;
    letter-spacing: -.8px;
}

.logo-text span {
    color: #252F29;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 22px;
}

.nav-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
}

.nav-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 30px;
    background: var(--green-light);
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .05em;
}

.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #35A66B;
}

/* HERO */

.hero {
    padding: 48px 0 32px;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    align-items: center;
    gap: 46px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.eyebrow::before {
    content: "";
    width: 27px;
    height: 2px;
    background: var(--gold);
}

.hero h1 {
    font: 800 clamp(38px, 4.8vw, 61px)/1.12 "Manrope", sans-serif;
    letter-spacing: -2.7px;
    color: #1B3022;
    margin: 19px 0;
}

.hero h1 em {
    color: var(--green);
    font-style: normal;
}

.hero-description {
    max-width: 470px;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.9;
    margin: 0;
}

.hero-actions {
    display: flex;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
    margin-top: 27px;
}

.button-primary {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    min-height: 51px;
    padding: 0 23px;
    border: 0;
    border-radius: 8px;
    background: var(--green);
    color: white;
    text-decoration: none;
    font-weight: 800;
    font-size: 13px;
    cursor: pointer;
    transition: background .2s, transform .2s, box-shadow .2s;
    box-shadow: 0 8px 20px rgba(24,95,53,.16);
}

.button-primary:hover {
    background: var(--green-dark);
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(24,95,53,.22);
}

.button-primary:disabled {
    opacity: .65;
    cursor: wait;
    transform: none;
}

.hero-caption {
    color: var(--muted);
    font-size: 12px;
}

/* HERO IMAGE */

.hero-visual {
    position: relative;
    min-width: 0;
    animation: enterRight .8s ease both;
}

.hero-image {
    height: 385px;
    position: relative;
    overflow: hidden;
    border-radius: 15px;
    background: #E3E9E1;
    box-shadow: 0 22px 55px rgba(24,55,35,.14);
}

.hero-image img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform .7s ease;
}

.hero-image:hover img {
    transform: scale(1.045);
}

.image-shade {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        180deg,
        transparent 50%,
        rgba(11,37,22,.55) 100%
    );
    pointer-events: none;
}

.image-label {
    position: absolute;
    left: 20px;
    bottom: 20px;
    color: white;
    z-index: 2;
}

.image-label small {
    display: block;
    color: #E0E8E0;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.image-label strong {
    display: block;
    margin-top: 4px;
    font: 700 21px "Manrope", sans-serif;
}

.floating-card {
    position: absolute;
    right: -17px;
    top: 25px;
    padding: 15px 17px;
    background: white;
    border: 1px solid rgba(24,95,53,.08);
    border-radius: 10px;
    box-shadow: 0 12px 35px rgba(0,0,0,.09);
    animation: float 4s ease-in-out infinite;
}

.floating-card small {
    display: block;
    font-size: 10px;
    color: var(--muted);
}

.floating-card strong {
    display: block;
    font: 800 14px "Manrope", sans-serif;
    color: var(--green);
    margin-top: 3px;
}

/* TRUST STRIP */

.trust-strip {
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    background: #FFFFFF;
}

.trust-inner {
    min-height: 84px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    align-items: center;
    gap: 20px;
}

.trust-item {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 11px;
}

.trust-icon {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    background: var(--green-light);
    color: var(--green);
    font-size: 17px;
}

.trust-item strong {
    display: block;
    font-size: 12px;
}

.trust-item span {
    display: block;
    color: var(--muted);
    font-size: 10px;
}

/* CALCULATOR SECTION */

.calculator-section {
    padding: 76px 0 85px;
    scroll-margin-top: 100px;
}

.section-heading {
    text-align: center;
    max-width: 620px;
    margin: 0 auto 34px;
}

.section-heading .eyebrow {
    justify-content: center;
}

.section-heading h2 {
    font: 800 clamp(28px, 3.5vw, 40px)/1.25 "Manrope", sans-serif;
    letter-spacing: -1.3px;
    margin: 13px 0;
    color: #1B3022;
}

.section-heading p {
    color: var(--muted);
    font-size: 14px;
    margin: 0;
}

.calculator-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
    gap: 24px;
    align-items: start;
}

.form-card,
.info-card {
    border: 1px solid var(--border);
    border-radius: 13px;
    background: white;
    box-shadow: var(--shadow);
}

.form-card {
    padding: 29px;
}

.card-heading {
    display: flex;
    gap: 13px;
    align-items: center;
    padding-bottom: 22px;
    margin-bottom: 23px;
    border-bottom: 1px solid var(--border);
}

.card-heading-icon {
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    color: var(--green);
    background: var(--green-light);
    font-size: 22px;
}

.card-heading h3 {
    font: 800 18px "Manrope", sans-serif;
    margin: 0;
}

.card-heading p {
    color: var(--muted);
    font-size: 11px;
    margin: 3px 0 0;
}

.field {
    margin-bottom: 19px;
}

.field label {
    display: block;
    color: #303C34;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 8px;
}

.field label .required {
    color: #BA6D39;
}

.input,
.select {
    display: block;
    width: 100%;
    min-height: 48px;
    padding: 11px 13px;
    border: 1px solid #DCE3DD;
    border-radius: 7px;
    outline: none;
    background: #FFFFFF;
    color: var(--ink);
    font-size: 13px;
    transition: border .2s, box-shadow .2s;
}

.input:focus,
.select:focus {
    border-color: var(--green);
    box-shadow: 0 0 0 3px rgba(24,95,53,.09);
}

.input::placeholder {
    color: #A1AAA3;
}

.input[readonly] {
    background: #F7F9F7;
    color: #637168;
}

.location-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.two-columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}

.hint {
    margin-top: 7px;
    color: var(--muted);
    font-size: 10px;
}

.status {
    display: none;
    margin-top: 9px;
    padding: 10px 12px;
    border-radius: 7px;
    font-size: 11px;
    overflow-wrap: anywhere;
}

.status.show {
    display: block;
}

.status.warn {
    color: #805A20;
    background: #FFF7E9;
}

.status.good {
    color: #17633C;
    background: #EDF7EF;
}

.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 9px;
}

.suggestion {
    border: 1px solid #D3E5D6;
    border-radius: 6px;
    padding: 7px 10px;
    color: var(--green);
    background: #F6FAF6;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    transition: background .2s, transform .2s;
}

.suggestion:hover {
    background: #E6F2E8;
    transform: translateY(-1px);
}

.submit-button {
    width: 100%;
    margin-top: 3px;
    min-height: 53px;
}

.form-note {
    margin: 13px 0 0;
    text-align: center;
    color: var(--muted);
    font-size: 10px;
}

/* INFORMATION CARD */

.info-card {
    overflow: hidden;
}

.info-top {
    padding: 26px;
    color: white;
    background: var(--green);
    position: relative;
    overflow: hidden;
}

.info-top::after {
    content: "⌂";
    position: absolute;
    right: 8px;
    top: -32px;
    color: rgba(255,255,255,.07);
    font-size: 145px;
    line-height: 1;
    pointer-events: none;
}

.info-top small {
    position: relative;
    z-index: 1;
    color: #C7E2CD;
    font-size: 10px;
    letter-spacing: .14em;
    font-weight: 800;
    text-transform: uppercase;
}

.info-top h3 {
    position: relative;
    z-index: 1;
    margin: 12px 0;
    font: 800 24px/1.35 "Manrope", sans-serif;
    letter-spacing: -.6px;
}

.info-top p {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #D9E9DC;
    font-size: 12px;
    line-height: 1.8;
}

.info-list {
    padding: 23px 25px 8px;
}

.info-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 22px;
}

.info-check {
    display: grid;
    place-items: center;
    flex: 0 0 31px;
    height: 31px;
    border-radius: 8px;
    background: var(--green-light);
    color: var(--green);
    font-size: 14px;
    font-weight: 800;
}

.info-row strong {
    display: block;
    color: #25342A;
    font-size: 12px;
}

.info-row p {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.7;
}

.info-disclaimer {
    margin: 0 23px 23px;
    padding: 13px;
    background: #F7F8F4;
    border: 1px solid var(--border);
    border-radius: 7px;
    color: #727B73;
    font-size: 10px;
    line-height: 1.8;
}

/* RESULT */

.result {
    display: none;
    margin-top: 22px;
    border: 1px solid #CFE3D1;
    border-radius: 11px;
    background: #F6FBF6;
    overflow: hidden;
}

.result.show {
    display: block;
    animation: fadeUp .4s ease both;
}

.result-top {
    padding: 21px 22px;
    border-bottom: 1px solid #DDEADE;
}

.result-label {
    font-size: 10px;
    color: var(--green);
    text-transform: uppercase;
    letter-spacing: .13em;
    font-weight: 800;
}

.result-amount {
    margin-top: 8px;
    color: #183D26;
    font: 800 clamp(28px, 4vw, 39px)/1.2 "Manrope", sans-serif;
    letter-spacing: -1.3px;
    overflow-wrap: anywhere;
}

.result-lakh {
    margin-top: 7px;
    font-size: 12px;
    color: #5B7560;
}

.result-disclaimer {
    padding: 14px 22px;
    margin: 0;
    color: #69786C;
    font-size: 10px;
    line-height: 1.8;
}

.error {
    display: none;
    margin-top: 14px;
    padding: 12px 14px;
    color: #9D3131;
    background: #FFF0EF;
    border: 1px solid #F4D6D3;
    border-radius: 7px;
    font-size: 12px;
    overflow-wrap: anywhere;
}

.error.show {
    display: block;
}

/* FOOTER */

.footer {
    background: #153D27;
    color: white;
    padding: 34px 0 23px;
}

.footer-main {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 25px;
    padding-bottom: 23px;
    border-bottom: 1px solid rgba(255,255,255,.13);
}

.footer-brand {
    font: 800 19px "Manrope", sans-serif;
}

.footer-brand span {
    color: #A9D2B2;
}

.footer-copy {
    color: #C5D5C9;
    font-size: 11px;
}

.footer-bottom {
    padding-top: 18px;
    color: #AFC3B4;
    font-size: 10px;
    line-height: 1.8;
}

/* ANIMATIONS */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes enterRight {
    from {
        opacity: 0;
        transform: translateX(16px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-6px);
    }
}

/* RESPONSIVE */

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr 1fr;
        gap: 25px;
    }

    .hero-image {
        height: 330px;
    }

    .calculator-layout {
        grid-template-columns: 1fr;
    }

    .info-card {
        display: grid;
        grid-template-columns: 1fr 1fr;
    }

    .info-top {
        grid-row: span 2;
    }

    .info-list {
        padding-bottom: 0;
    }

    .info-disclaimer {
        margin-top: 0;
        align-self: start;
    }
}

@media (max-width: 650px) {
    .container {
        width: min(100% - 30px, 520px);
    }

    .nav-inner {
        min-height: 68px;
    }

    .logo-text {
        font-size: 18px;
    }

    .logo-icon {
        width: 38px;
        height: 38px;
    }

    .nav-label {
        display: none;
    }

    .nav-badge {
        padding: 8px 10px;
        font-size: 9px;
    }

    .hero {
        padding-top: 34px;
    }

    .hero-grid {
        grid-template-columns: 1fr;
        gap: 29px;
    }

    .hero h1 {
        font-size: clamp(37px, 10vw, 51px);
        letter-spacing: -1.9px;
    }

    .hero-description {
        font-size: 13px;
    }

    .hero-image {
        height: 280px;
    }

    .floating-card {
        right: 10px;
        top: 10px;
    }

    .trust-inner {
        grid-template-columns: 1fr;
        gap: 0;
        padding: 12px 0;
    }

    .trust-item {
        justify-content: flex-start;
        padding: 10px 0;
    }

    .calculator-section {
        padding: 55px 0 60px;
    }

    .section-heading h2 {
        font-size: 30px;
    }

    .section-heading p {
        font-size: 12px;
    }

    .form-card {
        padding: 20px 16px;
    }

    .location-grid {
        grid-template-columns: 1fr;
    }

    .two-columns {
        gap: 10px;
    }

    .info-card {
        display: block;
    }

    .info-top {
        padding: 23px;
    }

    .info-top h3 {
        font-size: 22px;
    }

    .info-list {
        padding: 23px 21px 3px;
    }

    .info-disclaimer {
        margin: 0 20px 20px;
    }

    .footer-main {
        align-items: flex-start;
        flex-direction: column;
        gap: 10px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}
</style>
</head>

<body>

<header class="navbar">
    <div class="container nav-inner">

        <a href="/" class="logo">
            <span class="logo-icon">⌂</span>
            <span class="logo-text">BD<span>Properties</span></span>
        </a>

        <div class="nav-right">
            <span class="nav-label">PROPERTY PRICE ESTIMATOR</span>

            <span class="nav-badge">
                <span class="live-dot"></span>
                SMART ESTIMATES
            </span>
        </div>

    </div>
</header>

<main>

    <section class="hero">
        <div class="container hero-grid">

            <div class="hero-content">

                <div class="eyebrow">
                    Real estate, made simpler
                </div>

                <h1>
                    Find the value.<br>
                    <em>Know your place.</em>
                </h1>

                <p class="hero-description">
                    Make your next property decision with greater
                    confidence. Explore an indicative apartment
                    price based on location and property details.
                </p>

                <div class="hero-actions">
                    <a href="#calculator" class="button-primary">
                        Estimate property price
                        <span>→</span>
                    </a>

                    <span class="hero-caption">
                        Simple. Clear. Data-informed.
                    </span>
                </div>

            </div>

            <div class="hero-visual">

                <div class="hero-image">
                    <img
                        src="https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=1300&q=85"
                        alt="Contemporary residential property"
                        fetchpriority="high"
                    >

                    <div class="image-shade"></div>

                    <div class="image-label">
                        <small>Find your next home</small>
                        <strong>A better-informed move.</strong>
                    </div>
                </div>

                <div class="floating-card">
                    <small>Built for property decisions</small>
                    <strong>BDProperties</strong>
                </div>

            </div>

        </div>
    </section>

    <section class="trust-strip">
        <div class="container trust-inner">

            <div class="trust-item">
                <div class="trust-icon">⌖</div>
                <div>
                    <strong>Location-aware</strong>
                    <span>Supported areas</span>
                </div>
            </div>

            <div class="trust-item">
                <div class="trust-icon">▦</div>
                <div>
                    <strong>Property-specific</strong>
                    <span>Size and room details</span>
                </div>
            </div>

            <div class="trust-item">
                <div class="trust-icon">৳</div>
                <div>
                    <strong>Bangladeshi Taka</strong>
                    <span>Clear price estimates</span>
                </div>
            </div>

        </div>
    </section>

    <section class="calculator-section" id="calculator">

        <div class="container">

            <div class="section-heading">

                <div class="eyebrow">
                    Property price calculator
                </div>

                <h2>
                    Let's estimate your apartment.
                </h2>

                <p>
                    Enter your property details below to receive
                    an indicative price estimate.
                </p>

            </div>

            <div class="calculator-layout">

                <section class="form-card">

                    <div class="card-heading">

                        <div class="card-heading-icon">⌂</div>

                        <div>
                            <h3>Apartment information</h3>
                            <p>
                                Complete the fields to calculate
                                your estimate.
                            </p>
                        </div>

                    </div>

                    <form id="predictForm">

                        <div class="field">

                            <label for="locationSearch">
                                Property location
                                <span class="required">*</span>
                            </label>

                            <div class="location-grid">

                                <input
                                    class="input"
                                    id="locationSearch"
                                    type="search"
                                    placeholder="Search an area..."
                                    autocomplete="off"
                                    required
                                >

                                <select
                                    class="select"
                                    id="location"
                                    required
                                    aria-label="Choose supported location"
                                >
                                    <option value="">
                                        Choose location
                                    </option>
                                </select>

                            </div>

                            <div class="hint">
                                Search and choose a location from
                                the supported dataset.
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

                        <div class="two-columns">

                            <div class="field">
                                <label for="region">Region</label>

                                <input
                                    class="input"
                                    id="region"
                                    readonly
                                    placeholder="Auto-filled"
                                >
                            </div>

                            <div class="field">
                                <label for="sub_region">Sub-region</label>

                                <input
                                    class="input"
                                    id="sub_region"
                                    readonly
                                    placeholder="Auto-filled"
                                >
                            </div>

                        </div>

                        <div class="field">

                            <label for="area_sqft">
                                Apartment area (sq ft)
                                <span class="required">*</span>
                            </label>

                            <input
                                class="input"
                                id="area_sqft"
                                type="number"
                                min="100"
                                max="100000"
                                step="1"
                                placeholder="e.g. 1200"
                                required
                            >

                        </div>

                        <div class="two-columns">

                            <div class="field">

                                <label for="bedrooms">
                                    Bedrooms
                                    <span class="required">*</span>
                                </label>

                                <select
                                    class="select"
                                    id="bedrooms"
                                    required
                                >
                                    <option value="">Choose</option>
                                    <option value="1">1 bedroom</option>
                                    <option value="2">2 bedrooms</option>
                                    <option value="3" selected>3 bedrooms</option>
                                    <option value="4">4 bedrooms</option>
                                    <option value="5">5 bedrooms</option>
                                    <option value="6">6 bedrooms</option>
                                    <option value="7">7 bedrooms</option>
                                    <option value="8">8 bedrooms</option>
                                </select>

                            </div>

                            <div class="field">

                                <label for="bathrooms">
                                    Bathrooms
                                    <span class="required">*</span>
                                </label>

                                <select
                                    class="select"
                                    id="bathrooms"
                                    required
                                >
                                    <option value="">Choose</option>
                                    <option value="1">1 bathroom</option>
                                    <option value="2">2 bathrooms</option>
                                    <option value="3" selected>3 bathrooms</option>
                                    <option value="4">4 bathrooms</option>
                                    <option value="5">5 bathrooms</option>
                                    <option value="6">6 bathrooms</option>
                                    <option value="7">7 bathrooms</option>
                                    <option value="8">8 bathrooms</option>
                                </select>

                            </div>

                        </div>

                        <button
                            class="button-primary submit-button"
                            id="submitBtn"
                            type="submit"
                        >
                            Calculate estimated price
                            <span>→</span>
                        </button>

                        <p class="form-note">
                            No account required · Estimate in BDT
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

                        <div class="result-top">

                            <div class="result-label">
                                Estimated apartment price
                            </div>

                            <div
                                class="result-amount"
                                id="priceBdt"
                            >
                                ৳—
                            </div>

                            <div
                                class="result-lakh"
                                id="priceLakh"
                            ></div>

                        </div>

                        <p
                            class="result-disclaimer"
                            id="disclaimer"
                        ></p>

                    </section>

                </section>

                <aside class="info-card">

                    <div class="info-top">

                        <small>Designed for smarter decisions</small>

                        <h3>
                            Understand the estimate.
                            Plan your next move.
                        </h3>

                        <p>
                            A straightforward way to explore an
                            indicative apartment price using
                            the available property data.
                        </p>

                    </div>

                    <div class="info-list">

                        <div class="info-row">

                            <div class="info-check">✓</div>

                            <div>
                                <strong>Choose your location</strong>

                                <p>
                                    Select an area available
                                    in the current model dataset.
                                </p>
                            </div>

                        </div>

                        <div class="info-row">

                            <div class="info-check">✓</div>

                            <div>
                                <strong>Add apartment details</strong>

                                <p>
                                    Include the floor area,
                                    bedroom count and bathroom count.
                                </p>
                            </div>

                        </div>

                        <div class="info-row">

                            <div class="info-check">✓</div>

                            <div>
                                <strong>Review your estimate</strong>

                                <p>
                                    See the predicted price in
                                    Bangladeshi Taka and lakh.
                                </p>
                            </div>

                        </div>

                    </div>

                    <div class="info-disclaimer">
                        <strong>Important:</strong>
                        The result is an indicative estimate based
                        on historical listing data. It is not a
                        formal valuation, verified transaction price
                        or guarantee of the actual market price.
                    </div>

                </aside>

            </div>

        </div>

    </section>

</main>

<footer class="footer">
    <div class="container">

        <div class="footer-main">

            <div class="footer-brand">
                BD<span>Properties</span>
            </div>

            <div class="footer-copy">
                A clearer starting point for property decisions.
            </div>

        </div>

        <div class="footer-bottom">
            © <span id="year"></span> BDProperties.
            All rights reserved. Estimates are indicative only
            and may differ from actual market prices.
        </div>

    </div>
</footer>

<script>
const locationSelect = document.getElementById("location");
const locationSearch = document.getElementById("locationSearch");
const regionInput = document.getElementById("region");
const subRegionInput = document.getElementById("sub_region");

const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const errorBox = document.getElementById("errorBox");
const resultBox = document.getElementById("resultBox");

const locationStatus = document.getElementById("locationStatus");
const suggestions = document.getElementById("suggestions");

let locations = [];
let selectedLocation = null;
let searchTimer = null;

document.getElementById("year").textContent =
    new Date().getFullYear();


function showStatus(message, type = "warn") {
    locationStatus.textContent = message;
    locationStatus.className = "status show " + type;
}


function clearStatus() {
    locationStatus.textContent = "";
    locationStatus.className = "status";
}


function chooseLocation(item) {
    if (!item) return;

    selectedLocation = item;

    locationSelect.value = item.location;
    locationSearch.value = item.location;

    regionInput.value = item.region || "";
    subRegionInput.value = item.sub_region || "";

    suggestions.replaceChildren();

    showStatus(
        "Selected location: " + item.location,
        "good"
    );

    errorBox.classList.remove("show");
    resultBox.classList.remove("show");
}


function showSuggestions(items, exact = false) {
    suggestions.replaceChildren();

    if (!items || items.length === 0) {
        showStatus(
            "No matching supported location found. Try another area."
        );
        return;
    }

    if (!exact) {
        showStatus(
            "Choose the closest matching location. Name similarity does not indicate geographic distance."
        );
    } else {
        clearStatus();
    }

    items.slice(0, 6).forEach(item => {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "suggestion";
        button.textContent = item.location;

        button.addEventListener("click", () => {
            chooseLocation(item);
        });

        suggestions.appendChild(button);
    });
}


/* Load supported locations */

fetch("/locations.json")
    .then(response => {
        if (!response.ok) {
            throw new Error("Unable to load locations");
        }

        return response.json();
    })
    .then(data => {
        locations = data;

        locationSelect.innerHTML =
            '<option value="">Choose location</option>';

        locations.forEach(item => {
            const option = document.createElement("option");

            option.value = item.location;
            option.textContent = item.location;

            locationSelect.appendChild(option);
        });
    })
    .catch(() => {
        showStatus(
            "Unable to load locations. Please refresh the page."
        );
    });


/* Dropdown selection */

locationSelect.addEventListener("change", () => {
    const item = locations.find(
        entry => entry.location === locationSelect.value
    );

    if (item) {
        chooseLocation(item);
    } else {
        selectedLocation = null;
        regionInput.value = "";
        subRegionInput.value = "";
    }
});


/* Search supported locations */

locationSearch.addEventListener("input", () => {
    const query = locationSearch.value.trim();

    selectedLocation = null;
    locationSelect.value = "";
    regionInput.value = "";
    subRegionInput.value = "";

    suggestions.replaceChildren();
    errorBox.classList.remove("show");
    resultBox.classList.remove("show");

    clearTimeout(searchTimer);

    if (!query) {
        clearStatus();
        return;
    }

    const directMatches = locations.filter(item =>
        item.location.toLowerCase().includes(query.toLowerCase())
    ).slice(0, 6);

    if (directMatches.length) {
        showSuggestions(directMatches, true);
    }

    searchTimer = setTimeout(async () => {
        try {
            const response = await fetch(
                "/locations/search?q=" + encodeURIComponent(query)
            );

            if (!response.ok) {
                throw new Error("Search request failed");
            }

            const data = await response.json();

            if (locationSearch.value.trim() !== query) {
                return;
            }

            if (data.exact) {
                chooseLocation(data.exact);
                return;
            }

            showSuggestions(data.matches || [], false);

        } catch (error) {
            if (!directMatches.length) {
                showStatus(
                    "Location search is unavailable. You can use the dropdown instead."
                );
            }
        }
    }, 250);
});


/* Price prediction */

form.addEventListener("submit", async event => {
    event.preventDefault();

    errorBox.classList.remove("show");
    resultBox.classList.remove("show");

    if (
        !selectedLocation ||
        locationSelect.value !== selectedLocation.location
    ) {
        errorBox.textContent =
            "Please choose a location from the available suggestions or dropdown.";

        errorBox.classList.add("show");
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

    if (!Number.isFinite(area) || area < 100 || area > 100000) {
        errorBox.textContent =
            "Enter an apartment area between 100 and 100000 square feet.";

        errorBox.classList.add("show");
        return;
    }

    const payload = {
        location: selectedLocation.location,
        region: selectedLocation.region,
        sub_region: selectedLocation.sub_region,
        area_sqft: area,
        bedrooms: bedrooms,
        bathrooms: bathrooms
    };

    submitBtn.disabled = true;
    submitBtn.textContent = "Calculating estimate...";

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
            const message =
                typeof data.detail === "string"
                    ? data.detail
                    : "Prediction failed. Please check the details.";

            throw new Error(message);
        }

        const price = Number(data.estimated_price_bdt);
        const lakh = Number(data.estimated_price_lakh_bdt);

        document.getElementById("priceBdt").textContent =
            new Intl.NumberFormat("en-BD", {
                style: "currency",
                currency: "BDT",
                maximumFractionDigits: 0
            }).format(price);

        document.getElementById("priceLakh").textContent =
            "Approximately " +
            lakh.toLocaleString("en-BD", {
                maximumFractionDigits: 2
            }) +
            " lakh taka";

        document.getElementById("disclaimer").textContent =
            data.disclaimer ||
            "This is an indicative estimate, not a guaranteed market price.";

        resultBox.classList.add("show");

        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    } catch (error) {
        errorBox.textContent =
            error.message || "Something went wrong. Please try again.";

        errorBox.classList.add("show");

    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML =
            'Calculate estimated price <span>→</span>';
    }
});
</script>

</body>
</html>'''