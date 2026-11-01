from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "property_price_model.cbm"
METADATA_PATH = BASE_DIR / "model_metadata.json"

app = FastAPI(
    title="Property Price Predictor",
    version="3.0"
)

model = CatBoostRegressor()
metadata = {}
model_ready = False

if MODEL_PATH.exists() and METADATA_PATH.exists():
    model.load_model(str(MODEL_PATH))
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
    model_ready = True

FEATURES = [
    "Type",
    "Location",
    "No. Beds",
    "No. Baths",
    "Area",
    "Latitude",
    "Longitude",
    "Region",
    "Sub-region",
]


def money_bdt(value: float) -> str:
    return "৳" + f"{round(value):,}"


def location_options() -> str:
    locations = metadata.get("locations", [])
    output = []

    for row in locations:
        name = str(row.get("Location") or "").strip()

        if name:
            safe_name = html.escape(name, quote=True)
            output.append(
                f'<option value="{safe_name}"></option>'
            )

    return "\n".join(output)


def type_options(selected: str = "Apartment") -> str:
    types = metadata.get("types", ["Apartment"])
    output = []

    for value in types:
        value = str(value)
        safe_value = html.escape(value, quote=True)
        selected_attr = (
            " selected" if value == selected else ""
        )

        output.append(
            f'<option value="{safe_value}"{selected_attr}>'
            f'{safe_value}</option>'
        )

    return "\n".join(output)


def page(
    error: str = "",
    result: Optional[float] = None,
    form: Optional[dict] = None,
) -> str:

    form = form or {}

    location = str(form.get("location", ""))
    prop_type = str(
        form.get("property_type", "Apartment")
    )
    beds = form.get("bedrooms", "3")
    baths = form.get("bathrooms", "2")
    area = form.get("area_sqft", "1200")

    disabled = "" if model_ready else " disabled"

    if not model_ready:
        error = (
            "Model files are missing. "
            "Please ensure the model files are deployed."
        )

    error_html = ""

    if error:
        error_html = f"""
        <div class="alert error">
            {html.escape(error)}
        </div>
        """

    if result is not None:
        result_html = f"""
        <div class="result-content">
            <div class="result-icon">✓</div>

            <div class="result-label">
                ESTIMATED PROPERTY PRICE
            </div>

            <div class="price">
                {money_bdt(result)}
            </div>

            <div class="price-lakh">
                Approximately ৳{result / 100000:.2f} lakh
            </div>

            <div class="result-divider"></div>

            <p class="result-description">
                This is the estimated listing price based
                on the property details you entered.
            </p>

            <div class="result-badge">
                AI-Powered Estimate
            </div>

            <p class="disclaimer">
                This estimate is not a confirmed market
                price or a verified transaction value.
            </p>
        </div>
        """
    else:
        result_html = """
        <div class="empty-result">
            <div class="empty-icon">৳</div>

            <h3>Your estimated price will appear here</h3>

            <p>
                Enter the property details and click
                the prediction button to see the
                estimated property price.
            </p>

            <div class="empty-steps">
                <div>
                    <span>01</span>
                    <p>Select a location</p>
                </div>

                <div>
                    <span>02</span>
                    <p>Enter property details</p>
                </div>

                <div>
                    <span>03</span>
                    <p>Get your estimate</p>
                </div>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Property Price Predictor | Bangladesh</title>

<meta
    name="description"
    content="Estimate property listing prices in Bangladesh using property details and machine learning."
>

<style>
:root {{
    --primary: #2457d6;
    --primary-dark: #1945b4;
    --ink: #17233c;
    --muted: #718096;
    --border: #e5eaf2;
    --background: #f5f7fc;
    --green: #13845b;
}}

* {{
    box-sizing: border-box;
}}

html {{
    scroll-behavior: smooth;
}}

body {{
    margin: 0;
    min-height: 100vh;
    font-family: Inter, -apple-system, BlinkMacSystemFont,
        "Segoe UI", Arial, sans-serif;
    color: var(--ink);
    background:
        radial-gradient(
            circle at top left,
            #e8efff 0,
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #f6f8fd,
            #ffffff 55%,
            #f0faf6
        );
}}

header {{
    width: 100%;
    padding: 20px 6%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    border-bottom: 1px solid #e9edf5;
    background: rgba(255, 255, 255, 0.85);
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 21px;
    font-weight: 850;
    letter-spacing: -0.7px;
}}

.brand-icon {{
    width: 39px;
    height: 39px;
    display: grid;
    place-items: center;
    color: white;
    background: var(--primary);
    border-radius: 12px;
    font-size: 23px;
}}

.brand-price {{
    color: var(--primary);
}}

.header-badge {{
    padding: 9px 14px;
    border-radius: 30px;
    background: #e8f8f0;
    color: #11754f;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.8px;
}}

main {{
    max-width: 1160px;
    margin: 0 auto;
    padding: 42px 24px 65px;
}}

.hero {{
    text-align: center;
    max-width: 760px;
    margin: 0 auto 35px;
}}

.eyebrow {{
    display: inline-block;
    padding: 8px 13px;
    background: #eaf0ff;
    color: var(--primary);
    border-radius: 30px;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

h1 {{
    margin: 17px 0 13px;
    font-size: clamp(32px, 5vw, 49px);
    line-height: 1.1;
    letter-spacing: -1.8px;
    font-weight: 850;
}}

.hero p {{
    max-width: 650px;
    margin: 0 auto;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.8;
}}

.alert {{
    padding: 14px 16px;
    margin: 0 0 20px;
    border-radius: 12px;
    font-size: 14px;
}}

.error {{
    color: #a12626;
    background: #fff0f0;
    border: 1px solid #ffd2d2;
}}

.layout {{
    display: grid;
    grid-template-columns: minmax(0, 1.08fr)
                           minmax(0, 0.92fr);
    align-items: start;
    gap: 25px;
}}

.card {{
    min-width: 0;
    padding: 28px;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 23px;
    box-shadow: 0 15px 45px rgba(30, 55, 100, 0.07);
}}

.card-heading {{
    display: flex;
    align-items: flex-start;
    gap: 13px;
    margin-bottom: 25px;
}}

.heading-icon {{
    flex-shrink: 0;
    width: 45px;
    height: 45px;
    display: grid;
    place-items: center;
    background: #edf2ff;
    color: var(--primary);
    border-radius: 13px;
    font-size: 22px;
}}

.card h2 {{
    margin: 1px 0 6px;
    font-size: 21px;
    letter-spacing: -0.5px;
}}

.hint {{
    margin: 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}}

.fields {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px 16px;
}}

.field {{
    min-width: 0;
}}

.field.full {{
    grid-column: 1 / -1;
}}

label {{
    display: block;
    margin-bottom: 9px;
    font-size: 13px;
    font-weight: 750;
    color: #344054;
}}

.required {{
    color: #e04444;
}}

.input-wrap {{
    position: relative;
}}

.input-icon {{
    position: absolute;
    top: 50%;
    left: 13px;
    transform: translateY(-50%);
    color: #7c8ba5;
    font-size: 16px;
    pointer-events: none;
}}

input,
select {{
    display: block;
    width: 100%;
    min-width: 0;
    min-height: 49px;
    padding: 12px 13px;
    border: 1px solid #dce3ef;
    border-radius: 11px;
    background: #ffffff;
    color: var(--ink);
    font: inherit;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
}}

input:focus,
select:focus {{
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(36, 87, 214, 0.10);
}}

.input-with-icon {{
    padding-left: 40px;
}}

input::placeholder {{
    color: #9aa6b8;
}}

.field-help {{
    margin: 7px 0 0;
    color: #8490a3;
    font-size: 11px;
    line-height: 1.5;
}}

.btn {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    width: 100%;
    min-height: 53px;
    margin-top: 25px;
    padding: 14px 18px;
    border: 0;
    border-radius: 12px;
    background: linear-gradient(
        135deg,
        var(--primary),
        #376de8
    );
    color: #ffffff;
    font-size: 14px;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 9px 22px rgba(36, 87, 214, 0.22);
    transition: transform 0.2s, background 0.2s;
}}

.btn:hover {{
    background: var(--primary-dark);
    transform: translateY(-1px);
}}

.btn:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}}

.form-footer {{
    margin-top: 15px;
    text-align: center;
    color: #8994a7;
    font-size: 11px;
    line-height: 1.6;
}}

.right-column {{
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-width: 0;
}}

.result-card {{
    min-width: 0;
    padding: 27px;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 23px;
    box-shadow: 0 15px 45px rgba(30, 55, 100, 0.07);
}}

.result-heading {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 20px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--border);
}}

.result-heading-icon {{
    width: 43px;
    height: 43px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    color: var(--green);
    background: #e8f8f0;
    border-radius: 12px;
    font-size: 22px;
}}

.result-heading h2 {{
    margin: 0 0 5px;
    font-size: 19px;
}}

.result-heading p {{
    margin: 0;
    color: var(--muted);
    font-size: 12px;
}}

.result-content {{
    padding: 22px 19px;
    text-align: center;
    border: 1px solid #ccebdc;
    border-radius: 18px;
    background: linear-gradient(
        150deg,
        #f0fff7,
        #ffffff
    );
}}

.result-icon {{
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    margin: 0 auto 16px;
    border-radius: 50%;
    background: #d9f7e8;
    color: #087b50;
    font-size: 23px;
    font-weight: 900;
}}

.result-label {{
    color: #247453;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 1.2px;
}}

.price {{
    margin: 13px 0 7px;
    color: #116b47;
    font-size: clamp(27px, 3.3vw, 38px);
    font-weight: 900;
    letter-spacing: -1.2px;
    overflow-wrap: anywhere;
}}

.price-lakh {{
    color: #287b58;
    font-size: 14px;
    font-weight: 750;
}}

.result-divider {{
    height: 1px;
    margin: 23px 0 17px;
    background: #d8eee2;
}}

.result-description {{
    margin: 0;
    color: #64766c;
    font-size: 12px;
    line-height: 1.7;
}}

.result-badge {{
    display: inline-block;
    margin-top: 17px;
    padding: 8px 12px;
    border-radius: 30px;
    background: #ddf7e9;
    color: #176e49;
    font-size: 11px;
    font-weight: 800;
}}

.disclaimer {{
    margin: 18px 0 0;
    color: #7a8b82;
    font-size: 10px;
    line-height: 1.6;
}}

.empty-result {{
    padding: 10px 4px 2px;
    text-align: center;
}}

.empty-icon {{
    width: 66px;
    height: 66px;
    display: grid;
    place-items: center;
    margin: 0 auto 18px;
    border: 1px dashed #b8c8e9;
    border-radius: 20px;
    background: #f2f6ff;
    color: var(--primary);
    font-size: 31px;
    font-weight: 850;
}}

.empty-result h3 {{
    margin: 0 0 10px;
    font-size: 18px;
    line-height: 1.5;
}}

.empty-result > p {{
    max-width: 320px;
    margin: 0 auto;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.8;
}}

.empty-steps {{
    margin-top: 25px;
    padding-top: 19px;
    border-top: 1px solid var(--border);
    text-align: left;
}}

.empty-steps > div {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
}}

.empty-steps span {{
    width: 32px;
    height: 32px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: #edf2ff;
    color: var(--primary);
    font-size: 11px;
    font-weight: 850;
}}

.empty-steps p {{
    margin: 0;
    color: #536177;
    font-size: 12px;
    font-weight: 650;
}}

.info-card {{
    padding: 23px;
    border-radius: 20px;
    background: #172641;
    color: #ffffff;
}}

.info-card h3 {{
    margin: 0 0 12px;
    font-size: 17px;
}}

.info-card > p {{
    margin: 0;
    color: #d0daea;
    font-size: 12px;
    line-height: 1.8;
}}

.info-list {{
    padding: 0;
    margin: 19px 0 0;
    list-style: none;
}}

.info-list li {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
    color: #e1e8f3;
    font-size: 12px;
    line-height: 1.7;
}}

.info-list li span {{
    color: #82dfb3;
    font-weight: 900;
}}

.info-note {{
    margin-top: 19px;
    padding-top: 16px;
    border-top: 1px solid #ffffff20;
    color: #b9c6d9;
    font-size: 10px;
    line-height: 1.8;
}}

footer {{
    padding: 0 20px 27px;
    text-align: center;
    color: #8a96a8;
    font-size: 11px;
    line-height: 1.8;
}}

@media (max-width: 800px) {{
    main {{
        padding: 30px 18px 45px;
    }}

    .layout {{
        grid-template-columns: 1fr;
    }}

    .right-column {{
        order: 2;
    }}

    .card,
    .result-card {{
        padding: 23px;
    }}

    .price {{
        font-size: clamp(28px, 7vw, 38px);
    }}
}}

@media (max-width: 460px) {{
    header {{
        padding: 15px;
    }}

    .brand {{
        font-size: 17px;
    }}

    .brand-icon {{
        width: 35px;
        height: 35px;
    }}

    .header-badge {{
        font-size: 9px;
        padding: 8px;
    }}

    main {{
        padding: 27px 13px 38px;
    }}

    .hero {{
        margin-bottom: 25px;
    }}

    h1 {{
        font-size: 33px;
    }}

    .hero p {{
        font-size: 13px;
    }}

    .card,
    .result-card {{
        padding: 19px;
        border-radius: 18px;
    }}

    .fields {{
        grid-template-columns: 1fr;
        gap: 17px;
    }}

    .field.full {{
        grid-column: auto;
    }}
}}
</style>
</head>

<body>

<header>
    <div class="brand">
        <div class="brand-icon">⌂</div>
        <div>
            Property<span class="brand-price">Price</span>
        </div>
    </div>

    <div class="header-badge">
        AI-POWERED ESTIMATE
    </div>
</header>

<main>

    <section class="hero">
        <div class="eyebrow">
            Bangladesh Property Estimator
        </div>

        <h1>
            Find Your Property's<br>
            Estimated Price
        </h1>

        <p>
            Discover an estimated property listing price
            using location, property type, size and room
            details.
        </p>
    </section>

    {error_html}

    <div class="layout">

        <!-- LEFT SIDE: PROPERTY FORM -->

        <section class="card">

            <div class="card-heading">
                <div class="heading-icon">⌂</div>

                <div>
                    <h2>Property Details</h2>

                    <p class="hint">
                        Enter the details to estimate the price.
                    </p>
                </div>
            </div>

            <form method="post" action="/predict">

                <div class="fields">

                    <div class="field full">
                        <label for="property_type">
                            Property Type
                            <span class="required">*</span>
                        </label>

                        <select
                            id="property_type"
                            name="property_type"
                            required
                        >
                            {type_options(prop_type)}
                        </select>
                    </div>

                    <div class="field full">
                        <label for="location">
                            Location
                            <span class="required">*</span>
                        </label>

                        <div class="input-wrap">
                            <span class="input-icon">⌕</span>

                            <input
                                id="location"
                                name="location"
                                type="text"
                                class="input-with-icon"
                                list="location-options"
                                placeholder="Search or select a location"
                                value="{html.escape(location, quote=True)}"
                                autocomplete="off"
                                required
                            >

                            <datalist id="location-options">
                                {location_options()}
                            </datalist>
                        </div>

                        <p class="field-help">
                            Type to search, then select a location
                            from the suggestions.
                        </p>
                    </div>

                    <div class="field full">
                        <label for="area_sqft">
                            Property Area (sq ft)
                            <span class="required">*</span>
                        </label>

                        <input
                            id="area_sqft"
                            name="area_sqft"
                            type="number"
                            min="100"
                            max="1000000"
                            step="1"
                            placeholder="Example: 1200"
                            value="{html.escape(str(area), quote=True)}"
                            required
                        >

                        <p class="field-help">
                            Enter the property's area in square feet.
                        </p>
                    </div>

                    <div class="field">
                        <label for="bedrooms">
                            Bedrooms
                        </label>

                        <input
                            id="bedrooms"
                            name="bedrooms"
                            type="number"
                            min="0"
                            max="30"
                            step="1"
                            placeholder="e.g. 3"
                            value="{html.escape(str(beds), quote=True)}"
                        >
                    </div>

                    <div class="field">
                        <label for="bathrooms">
                            Bathrooms
                        </label>

                        <input
                            id="bathrooms"
                            name="bathrooms"
                            type="number"
                            min="0"
                            max="30"
                            step="1"
                            placeholder="e.g. 2"
                            value="{html.escape(str(baths), quote=True)}"
                        >
                    </div>

                </div>

                <button
                    class="btn"
                    type="submit"
                    {disabled}
                >
                    Predict Property Price
                    <span>→</span>
                </button>

                <div class="form-footer">
                    Estimates are generated using the deployed
                    machine-learning model.
                </div>

            </form>

        </section>

        <!-- RIGHT SIDE: PREDICTION OUTPUT -->

        <div class="right-column">

            <section class="result-card">

                <div class="result-heading">
                    <div class="result-heading-icon">৳</div>

                    <div>
                        <h2>Price Estimate</h2>

                        <p>
                            Your estimated property value
                        </p>
                    </div>
                </div>

                {result_html}

            </section>

            <aside class="info-card">

                <h3>How It Works</h3>

                <p>
                    The prediction model learns patterns from
                    property listings and uses your inputs
                    to estimate a listing price.
                </p>

                <ul class="info-list">
                    <li>
                        <span>✓</span>
                        Choose a property type and location.
                    </li>

                    <li>
                        <span>✓</span>
                        Enter the area and room details.
                    </li>

                    <li>
                        <span>✓</span>
                        View your estimated price on the right.
                    </li>
                </ul>

                <div class="info-note">
                    Estimates are informational only.
                    Actual prices may depend on building condition,
                    floor, amenities, ownership documents and
                    market conditions. The dataset contains
                    asking/listing prices, not verified sale
                    transactions.
                </div>

            </aside>

        </div>

    </div>

</main>

<footer>
    Property Price Predictor · Bangladesh<br>
    Estimates are not a confirmed valuation or financial advice.
</footer>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return page()


@app.get("/health")
def health():
    return {
        "status": "ok" if model_ready else "model_missing",
        "model_loaded": model_ready,
    }


@app.post("/predict", response_class=HTMLResponse)
def predict(
    property_type: str = Form(...),
    location: str = Form(...),
    area_sqft: float = Form(...),
    bedrooms: Optional[float] = Form(None),
    bathrooms: Optional[float] = Form(None),
):
    form_data = {
        "property_type": property_type,
        "location": location,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms if bedrooms is not None else "",
        "bathrooms": bathrooms if bathrooms is not None else "",
    }

    if not model_ready:
        return HTMLResponse(
            page(
                "The model has not been trained or deployed.",
                form=form_data,
            ),
            status_code=503,
        )

    if not np.isfinite(area_sqft) or area_sqft <= 0 or area_sqft > 1_000_000:
        return HTMLResponse(
            page(
                "Please enter a valid property area in square feet.",
                form=form_data,
            ),
            status_code=400,
        )

    if bedrooms is not None and (
        not np.isfinite(bedrooms) or bedrooms < 0 or bedrooms > 30
    ):
        return HTMLResponse(
            page("Please enter a valid number of bedrooms.", form=form_data),
            status_code=400,
        )

    if bathrooms is not None and (
        not np.isfinite(bathrooms) or bathrooms < 0 or bathrooms > 30
    ):
        return HTMLResponse(
            page("Please enter a valid number of bathrooms.", form=form_data),
            status_code=400,
        )

    # Match the exact location in the model metadata.
    match = next(
        (
            row
            for row in metadata.get("locations", [])
            if str(row.get("Location") or "").strip() == location.strip()
        ),
        None,
    )

    if not match:
        return HTMLResponse(
            page(
                "Please select a valid location from the suggestions.",
                form=form_data,
            ),
            status_code=400,
        )

    row = {
        "Type": property_type,
        "Location": str(match.get("Location") or location),
        "No. Beds": (
            float(bedrooms)
            if bedrooms not in (None, "")
            else np.nan
        ),
        "No. Baths": (
            float(bathrooms)
            if bathrooms not in (None, "")
            else np.nan
        ),
        "Area": float(area_sqft),
        "Latitude": (
            match.get("Latitude")
            if match.get("Latitude") is not None
            else np.nan
        ),
        "Longitude": (
            match.get("Longitude")
            if match.get("Longitude") is not None
            else np.nan
        ),
        "Region": str(match.get("Region") or "Unknown"),
        "Sub-region": str(match.get("Sub-region") or "Unknown"),
    }

    frame = pd.DataFrame([row], columns=FEATURES)

    try:
        predicted = float(
            np.expm1(model.predict(frame)[0])
        )

        predicted = max(0.0, predicted)

    except Exception:
        return HTMLResponse(
            page(
                "Prediction failed. Please check the property details and try again.",
                form=form_data,
            ),
            status_code=500,
        )

    return HTMLResponse(
        page(
            result=predicted,
            form=form_data,
        )
    )


@app.post("/api/predict")
def api_predict(payload: dict):
    """JSON prediction endpoint for future integrations."""

    if not model_ready:
        return JSONResponse(
            {"error": "Model not loaded"},
            status_code=503,
        )

    try:
        location = str(payload["location"]).strip()

        match = next(
            (
                row
                for row in metadata.get("locations", [])
                if str(row.get("Location") or "").strip() == location
            ),
            None,
        )

        if not match:
            return JSONResponse(
                {"error": "Unknown location"},
                status_code=400,
            )

        area = float(payload["area_sqft"])

        if not np.isfinite(area) or area <= 0 or area > 1_000_000:
            return JSONResponse(
                {"error": "Invalid property area"},
                status_code=400,
            )

        bedrooms_value = payload.get("bedrooms")
        bathrooms_value = payload.get("bathrooms")

        bedrooms_value = (
            float(bedrooms_value)
            if bedrooms_value not in (None, "")
            else np.nan
        )

        bathrooms_value = (
            float(bathrooms_value)
            if bathrooms_value not in (None, "")
            else np.nan
        )

        row = {
            "Type": str(payload.get("type", "Apartment")),
            "Location": str(match.get("Location") or location),
            "No. Beds": bedrooms_value,
            "No. Baths": bathrooms_value,
            "Area": area,
            "Latitude": (
                match.get("Latitude")
                if match.get("Latitude") is not None
                else np.nan
            ),
            "Longitude": (
                match.get("Longitude")
                if match.get("Longitude") is not None
                else np.nan
            ),
            "Region": str(match.get("Region") or "Unknown"),
            "Sub-region": str(match.get("Sub-region") or "Unknown"),
        }

        frame = pd.DataFrame([row], columns=FEATURES)

        predicted = max(
            0.0,
            float(np.expm1(model.predict(frame)[0])),
        )

        return {
            "estimated_price_bdt": round(predicted),
            "estimated_price_lakh": round(predicted / 100000, 2),
            "note": (
                "Estimated listing price, not a verified "
                "transaction price."
            ),
        }

    except (KeyError, ValueError, TypeError) as exc:
        return JSONResponse(
            {"error": f"Invalid input: {exc}"},
            status_code=400,
        )