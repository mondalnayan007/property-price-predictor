def page(
    error: str = "",
    result: Optional[float] = None,
    form: Optional[dict] = None,
) -> str:

    form = form or {}

    location = str(form.get("location", ""))
    prop_type = str(form.get("property_type", "Apartment"))
    beds = form.get("bedrooms", "3")
    baths = form.get("bathrooms", "2")
    area = form.get("area_sqft", "1200")

    disabled = "" if model_ready else " disabled"

    if not model_ready:
        error = (
            "The prediction model is not available. "
            "Please check the deployed model files."
        )

    error_html = ""

    if error:
        error_html = f"""
        <div class="alert error">
            <span class="alert-icon">!</span>
            <span>{html.escape(error)}</span>
        </div>
        """

    # Prediction result card
    if result is not None:
        result_html = f"""
        <div class="result-success">

            <div class="success-icon">✓</div>

            <div class="result-overline">
                YOUR ESTIMATED PROPERTY VALUE
            </div>

            <div class="price">
                {money_bdt(result)}
            </div>

            <div class="price-lakh">
                ৳{result / 100000:.2f} Lakh
            </div>

            <div class="result-line"></div>

            <div class="result-message">
                Your estimated listing price is ready.
                This estimate is based on the details
                you entered and the available dataset.
            </div>

            <div class="estimate-tag">
                <span class="tag-dot"></span>
                ML-powered estimate
            </div>

            <div class="result-disclaimer">
                This is an estimated listing price,
                not a guaranteed sale price.
            </div>

        </div>
        """
    else:
        result_html = """
        <div class="result-empty">

            <div class="illustration">

                <div class="sun"></div>

                <div class="building building-back">
                    <div class="window-grid">
                        <i></i><i></i><i></i>
                        <i></i><i></i><i></i>
                    </div>
                </div>

                <div class="building building-main">
                    <div class="roof"></div>
                    <div class="window-grid">
                        <i></i><i></i>
                        <i></i><i></i>
                        <i></i><i></i>
                        <i></i><i></i>
                    </div>
                    <div class="building-door"></div>
                </div>

                <div class="building building-front">
                    <div class="window-grid">
                        <i></i><i></i>
                        <i></i><i></i>
                        <i></i><i></i>
                    </div>
                </div>

                <div class="floating-price">
                    <span>৳</span>
                    <div>
                        <small>PROPERTY VALUE</small>
                        <b>Estimate</b>
                    </div>
                </div>

                <div class="floating-check">✓</div>

                <div class="ground"></div>

            </div>

            <h3>Your property value starts here.</h3>

            <p class="empty-description">
                Add your property details to discover
                an estimated listing price.
            </p>

            <div class="steps">

                <div class="step">
                    <span class="step-number">01</span>
                    <div>
                        <b>Choose a location</b>
                        <small>Select your property area.</small>
                    </div>
                </div>

                <div class="step">
                    <span class="step-number">02</span>
                    <div>
                        <b>Enter property details</b>
                        <small>Add size and room information.</small>
                    </div>
                </div>

                <div class="step">
                    <span class="step-number">03</span>
                    <div>
                        <b>Get your estimate</b>
                        <small>View the predicted price here.</small>
                    </div>
                </div>

            </div>

        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<meta
    name="theme-color"
    content="#185F35"
>

<meta
    name="description"
    content="BDProperties helps estimate property listing prices in Bangladesh using machine learning."
>

<title>BDProperties | Property Price Estimator</title>

<style>

/* =========================================
   BDPROPERTIES — DESIGN SYSTEM
========================================= */

:root {{
    --primary: #185F35;
    --primary-dark: #104526;
    --primary-light: #287b49;
    --primary-soft: #eaf4ed;

    --ink: #172b20;
    --muted: #738176;
    --border: #e3ebe4;

    --surface: #ffffff;
    --background: #f5f8f5;

    --success: #16834a;
    --shadow: 0 18px 55px rgba(24, 95, 53, 0.08);
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

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    color: var(--ink);

    background:
        radial-gradient(
            circle at 8% 15%,
            rgba(24, 95, 53, 0.07),
            transparent 27%
        ),
        radial-gradient(
            circle at 90% 38%,
            rgba(100, 164, 111, 0.08),
            transparent 28%
        ),
        var(--background);

    overflow-x: hidden;
}}

button,
input,
select {{
    font: inherit;
}}

button,
a,
input,
select {{
    -webkit-tap-highlight-color: transparent;
}}

/* =========================================
   HEADER
========================================= */

.topbar {{
    position: relative;
    z-index: 10;

    width: 100%;

    padding: 19px 6%;

    display: flex;
    align-items: center;
    justify-content: space-between;

    gap: 18px;

    background: rgba(255, 255, 255, 0.91);

    border-bottom: 1px solid rgba(24, 95, 53, 0.09);

    backdrop-filter: blur(16px);
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 11px;

    text-decoration: none;
    color: var(--ink);
}}

.brand-mark {{
    position: relative;

    width: 43px;
    height: 43px;

    display: grid;
    place-items: center;

    border-radius: 13px;

    color: white;

    background: var(--primary);

    box-shadow:
        0 7px 18px rgba(24, 95, 53, 0.22);

    overflow: hidden;
}}

.brand-mark svg {{
    width: 25px;
    height: 25px;
}}

.brand-name {{
    font-size: 22px;
    font-weight: 850;
    letter-spacing: -0.8px;
}}

.brand-name span {{
    color: var(--primary);
}}

.header-right {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.header-status {{
    display: flex;
    align-items: center;
    gap: 8px;

    padding: 10px 14px;

    background: var(--primary-soft);

    border: 1px solid #d7e9db;

    border-radius: 40px;

    color: var(--primary);

    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.7px;
}}

.status-dot {{
    width: 8px;
    height: 8px;

    background: #26a269;
    border-radius: 50%;

    box-shadow: 0 0 0 4px rgba(38, 162, 105, 0.12);
}}

/* =========================================
   PAGE BACKGROUND DECORATION
========================================= */

.page-shell {{
    position: relative;
    isolation: isolate;
    overflow: hidden;
}}

.background-decoration {{
    position: absolute;
    inset: 0;

    z-index: -1;

    overflow: hidden;
    pointer-events: none;
}}

.decor-circle {{
    position: absolute;

    border-radius: 50%;

    border: 1px solid rgba(24, 95, 53, 0.08);
}}

.decor-circle.one {{
    width: 360px;
    height: 360px;

    top: 70px;
    left: -180px;

    animation: rotateSlow 35s linear infinite;
}}

.decor-circle.two {{
    width: 480px;
    height: 480px;

    top: 160px;
    right: -280px;

    animation: rotateSlow 45s linear infinite reverse;
}}

.decor-circle.three {{
    width: 240px;
    height: 240px;

    top: 400px;
    left: 40%;

    border-style: dashed;

    animation: rotateSlow 60s linear infinite;
}}

.bg-building {{
    position: absolute;
    bottom: 0;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-evenly;

    border: 1px solid rgba(24, 95, 53, 0.06);

    background: rgba(24, 95, 53, 0.025);

    opacity: 0.7;
}}

.bg-building::before {{
    content: "";

    position: absolute;
    top: 12px;
    left: 15%;
    right: 15%;

    height: 3px;

    border-radius: 5px;

    background: rgba(24, 95, 53, 0.10);
}}

.bg-building i {{
    display: block;

    width: 9px;
    height: 12px;

    border: 1px solid rgba(24, 95, 53, 0.10);

    border-radius: 2px;
}}

.bg-building .window-row {{
    width: 100%;

    display: flex;
    justify-content: space-evenly;
}}

.bg-building.a {{
    left: 1%;
    width: 80px;
    height: 180px;
}}

.bg-building.b {{
    left: 8%;
    width: 105px;
    height: 250px;
}}

.bg-building.c {{
    right: 3%;
    width: 100px;
    height: 220px;
}}

.bg-building.d {{
    right: 11%;
    width: 75px;
    height: 155px;
}}

.floating-particle {{
    position: absolute;

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #70a981;

    opacity: 0.5;

    animation: floatParticle 8s ease-in-out infinite;
}}

.particle-one {{
    top: 160px;
    left: 12%;
}}

.particle-two {{
    top: 320px;
    right: 12%;
    animation-delay: -2s;
}}

.particle-three {{
    top: 500px;
    left: 23%;
    animation-delay: -4s;
}}

.particle-four {{
    top: 220px;
    right: 25%;
    animation-delay: -1s;
}}

@keyframes rotateSlow {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}

@keyframes floatParticle {{
    0%, 100% {{
        transform: translateY(0);
        opacity: 0.3;
    }}

    50% {{
        transform: translateY(-22px);
        opacity: 0.8;
    }}
}}

/* =========================================
   MAIN CONTAINER
========================================= */

main {{
    position: relative;

    max-width: 1210px;

    margin: 0 auto;

    padding: 43px 25px 70px;
}}

.hero {{
    max-width: 790px;

    margin: 0 auto 39px;

    text-align: center;

    animation: fadeUp 0.65s ease both;
}}

.hero-label {{
    display: inline-flex;
    align-items: center;
    gap: 9px;

    padding: 9px 14px;

    border: 1px solid #d6e7d9;
    border-radius: 40px;

    background: rgba(255, 255, 255, 0.82);

    color: var(--primary);

    font-size: 11px;
    font-weight: 850;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}}

.hero-label svg {{
    width: 15px;
    height: 15px;
}}

.hero h1 {{
    margin: 20px 0 16px;

    font-size: clamp(35px, 5vw, 57px);

    line-height: 1.08;

    letter-spacing: -2.3px;

    font-weight: 900;
}}

.hero h1 .highlight {{
    color: var(--primary);
}}

.hero-description {{
    max-width: 640px;

    margin: 0 auto;

    color: var(--muted);

    font-size: 15px;
    line-height: 1.9;
}}

/* =========================================
   MAIN TWO-COLUMN LAYOUT
========================================= */

.layout {{
    display: grid;

    grid-template-columns:
        minmax(0, 1.04fr)
        minmax(0, 0.96fr);

    gap: 25px;

    align-items: start;
}}

.panel {{
    min-width: 0;

    background: rgba(255, 255, 255, 0.96);

    border: 1px solid rgba(24, 95, 53, 0.10);

    border-radius: 23px;

    box-shadow: var(--shadow);

    overflow: hidden;

    animation: fadeUp 0.7s ease both;
}}

.form-panel {{
    animation-delay: 0.08s;
}}

.result-panel {{
    animation-delay: 0.16s;
}}

.panel-header {{
    display: flex;
    align-items: center;
    gap: 14px;

    padding: 25px 27px 21px;

    border-bottom: 1px solid #edf1ed;
}}

.panel-icon {{
    flex-shrink: 0;

    width: 46px;
    height: 46px;

    display: grid;
    place-items: center;

    border-radius: 14px;

    background: var(--primary-soft);

    color: var(--primary);
}}

.panel-icon svg {{
    width: 24px;
    height: 24px;
}}

.panel-title {{
    margin: 0 0 5px;

    font-size: 19px;
    font-weight: 850;
    letter-spacing: -0.5px;
}}

.panel-subtitle {{
    margin: 0;

    color: var(--muted);

    font-size: 12px;
    line-height: 1.7;
}}

.panel-body {{
    padding: 26px 27px 28px;
}}

/* =========================================
   FORM FIELDS
========================================= */

.fields {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 21px 17px;
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

    color: #344c3a;

    font-size: 12px;
    font-weight: 800;
}}

.required {{
    color: #d94848;
}}

.input-container {{
    position: relative;
}}

.input-symbol {{
    position: absolute;

    left: 14px;
    top: 50%;

    transform: translateY(-50%);

    display: grid;
    place-items: center;

    color: #7a9380;

    pointer-events: none;
}}

.input-symbol svg {{
    width: 18px;
    height: 18px;
}}

input,
select {{
    display: block;

    width: 100%;
    min-width: 0;
    min-height: 50px;

    padding: 13px 14px;

    border: 1px solid #dce6dd;
    border-radius: 11px;

    outline: none;

    color: var(--ink);
    background: #ffffff;

    font-size: 13px;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease,
        background 0.2s ease;
}}

input.with-icon {{
    padding-left: 43px;
}}

input::placeholder {{
    color: #a1ada3;
}}

input:hover,
select:hover {{
    border-color: #a8c9b0;
}}

input:focus,
select:focus {{
    border-color: var(--primary);

    box-shadow: 0 0 0 4px rgba(24, 95, 53, 0.09);

    background: #ffffff;
}}

.field-help {{
    margin: 7px 0 0;

    color: #8a978c;

    font-size: 10px;
    line-height: 1.7;
}}

/* =========================================
   LOCATION SEARCH
========================================= */

.location-container {{
    position: relative;
}}

.location-container input {{
    padding-right: 42px;
}}

.search-mark {{
    position: absolute;

    top: 50%;
    right: 15px;

    transform: translateY(-50%);

    color: var(--primary);

    font-size: 20px;

    pointer-events: none;
}}

.location-tip {{
    display: flex;
    align-items: flex-start;
    gap: 7px;

    margin-top: 9px;

    color: #7d8d80;

    font-size: 10px;
    line-height: 1.7;
}}

.location-tip span {{
    color: var(--primary);
    font-weight: 900;
}}

/* =========================================
   PRIMARY BUTTON
========================================= */

.predict-button {{
    position: relative;

    display: flex;
    align-items: center;
    justify-content: center;
    gap: 11px;

    width: 100%;
    min-height: 54px;

    margin-top: 26px;
    padding: 15px 20px;

    border: 0;
    border-radius: 12px;

    color: #ffffff;

    background:
        linear-gradient(
            135deg,
            #185F35,
            #287b49
        );

    box-shadow:
        0 8px 19px rgba(24, 95, 53, 0.19);

    font-size: 13px;
    font-weight: 850;

    cursor: pointer;

    overflow: hidden;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        background 0.2s ease;
}}

.predict-button::before {{
    content: "";

    position: absolute;

    top: 0;
    left: -120%;

    width: 65%;
    height: 100%;

    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.20),
        transparent
    );

    transform: skewX(-20deg);

    transition: left 0.65s ease;
}}

.predict-button:hover::before {{
    left: 140%;
}}

.predict-button:hover {{
    transform: translateY(-2px);

    box-shadow:
        0 12px 26px rgba(24, 95, 53, 0.25);
}}

.predict-button:active {{
    transform: translateY(0);
}}

.predict-button:disabled {{
    opacity: 0.55;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
}}

.predict-button svg {{
    width: 17px;
    height: 17px;
}}

.form-footnote {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 7px;

    margin-top: 15px;

    color: #849186;

    font-size: 10px;
    line-height: 1.7;

    text-align: center;
}}

.form-footnote svg {{
    flex-shrink: 0;
    width: 14px;
    height: 14px;
    color: var(--primary);
}}

/* =========================================
   RESULT PANEL
========================================= */

.result-panel .panel-body {{
    padding: 24px;
}}

.result-success {{
    position: relative;

    padding: 27px 19px 23px;

    text-align: center;

    border: 1px solid #cce5d2;
    border-radius: 17px;

    background:
        radial-gradient(
            circle at top right,
            rgba(67, 151, 87, 0.09),
            transparent 40%
        ),
        linear-gradient(
            145deg,
            #f0faf2,
            #ffffff
        );

    animation: resultAppear 0.55s ease both;
}}

.success-icon {{
    display: grid;
    place-items: center;

    width: 46px;
    height: 46px;

    margin: 0 auto 17px;

    border-radius: 50%;

    background: #d9f1df;
    color: var(--primary);

    font-size: 22px;
    font-weight: 900;
}}

.result-overline {{
    color: #4a7656;

    font-size: 10px;
    font-weight: 900;

    letter-spacing: 1.5px;
}}

.price {{
    margin: 15px 0 7px;

    color: var(--primary);

    font-size: clamp(27px, 3.1vw, 39px);
    font-weight: 900;

    letter-spacing: -1.5px;

    overflow-wrap: anywhere;
}}

.price-lakh {{
    color: #507c5b;

    font-size: 14px;
    font-weight: 750;
}}

.result-line {{
    height: 1px;

    margin: 23px 0 18px;

    background: #dceade;
}}

.result-message {{
    max-width: 330px;

    margin: 0 auto;

    color: #718275;

    font-size: 12px;
    line-height: 1.9;
}}

.estimate-tag {{
    display: inline-flex;
    align-items: center;
    gap: 8px;

    margin-top: 19px;
    padding: 9px 13px;

    border-radius: 30px;

    background: #e0f2e4;
    color: var(--primary);

    font-size: 10px;
    font-weight: 850;
}}

.tag-dot {{
    width: 7px;
    height: 7px;

    background: #279254;
    border-radius: 50%;
}}

.result-disclaimer {{
    margin-top: 18px;

    color: #8a978c;

    font-size: 10px;
    line-height: 1.8;
}}

/* =========================================
   EMPTY RESULT / BUILDING ILLUSTRATION
========================================= */

.result-empty {{
    text-align: center;
}}

.illustration {{
    position: relative;

    width: 100%;
    max-width: 370px;

    height: 235px;

    margin: 0 auto 23px;

    border-radius: 17px;

    background:
        radial-gradient(
            circle at 52% 35%,
            #e1f1e4 0,
            #f0f7f1 43%,
            #f8fbf8 78%
        );

    overflow: hidden;
}}

.sun {{
    position: absolute;

    top: 31px;
    right: 65px;

    width: 43px;
    height: 43px;

    border-radius: 50%;

    background: #d7ebd9;

    animation: sunPulse 5s ease-in-out infinite;
}}

.building {{
    position: absolute;
    bottom: 35px;

    border: 1px solid rgba(24, 95, 53, 0.12);

    border-radius: 5px 5px 0 0;

    box-shadow: 0 7px 17px rgba(24, 95, 53, 0.05);
}}

.building-back {{
    left: 37px;

    width: 83px;
    height: 106px;

    background: #c9e1cd;

    animation: buildingFloat 5s ease-in-out infinite;
}}

.building-main {{
    z-index: 2;

    left: 50%;
    bottom: 35px;

    width: 102px;
    height: 151px;

    transform: translateX(-50%);

    background: #ffffff;

    border-color: #a6c8ac;

    animation: buildingFloat 5s ease-in-out infinite;
    animation-delay: -1.5s;
}}

.roof {{
    position: absolute;

    top: -7px;
    left: -5px;
    right: -5px;

    height: 8px;

    border-radius: 3px;

    background: var(--primary);
}}

.building-front {{
    right: 31px;

    width: 75px;
    height: 91px;

    background: #dcecdf;

    animation: buildingFloat 5s ease-in-out infinite;
    animation-delay: -2.4s;
}}

.window-grid {{
    position: absolute;

    top: 19px;
    left: 12px;
    right: 12px;

    display: grid;

    grid-template-columns: repeat(2, 1fr);

    gap: 9px;
}}

.window-grid i {{
    height: 14px;

    border: 1px solid #b2d2b7;

    border-radius: 3px;

    background: #e0f1e3;
}}

.building-back .window-grid,
.building-front .window-grid {{
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}}

.building-back .window-grid i,
.building-front .window-grid i {{
    height: 12px;
}}

.building-door {{
    position: absolute;

    bottom: 0;
    left: 50%;

    width: 23px;
    height: 34px;

    transform: translateX(-50%);

    border-radius: 5px 5px 0 0;

    background: var(--primary);
}}

.ground {{
    position: absolute;

    bottom: 26px;
    left: 7%;
    right: 7%;

    height: 2px;

    border-radius: 50%;

    background: #c2dec7;
}}

.floating-price {{
    position: absolute;
    z-index: 3;

    top: 35px;
    left: 17px;

    display: flex;
    align-items: center;
    gap: 10px;

    padding: 11px 13px;

    border: 1px solid #e2ebe3;
    border-radius: 12px;

    background: rgba(255, 255, 255, 0.96);

    box-shadow: 0 8px 22px rgba(24, 95, 53, 0.10);

    text-align: left;

    animation: floatingCard 4s ease-in-out infinite;
}}

.floating-price > span {{
    display: grid;
    place-items: center;

    width: 31px;
    height: 31px;

    border-radius: 9px;

    background: var(--primary-soft);

    color: var(--primary);

    font-size: 18px;
    font-weight: 900;
}}

.floating-price small {{
    display: block;

    margin-bottom: 4px;

    color: #8b9a8d;

    font-size: 7px;
    font-weight: 850;
    letter-spacing: 0.6px;
}}

.floating-price b {{
    display: block;

    color: var(--primary);

    font-size: 12px;
}}

.floating-check {{
    position: absolute;
    z-index: 4;

    right: 25px;
    bottom: 75px;

    display: grid;
    place-items: center;

    width: 33px;
    height: 33px;

    border: 3px solid #ffffff;
    border-radius: 50%;

    background: #278b4b;

    color: #ffffff;

    font-size: 14px;
    font-weight: 900;

    box-shadow: 0 5px 12px rgba(24, 95, 53, 0.14);

    animation: checkPulse 3s ease-in-out infinite;
}}

.result-empty h3 {{
    margin: 0 0 10px;

    color: var(--ink);

    font-size: 19px;
    font-weight: 850;

    letter-spacing: -0.5px;
}}

.empty-description {{
    max-width: 350px;

    margin: 0 auto;

    color: var(--muted);

    font-size: 12px;
    line-height: 1.9;
}}

/* =========================================
   STEPS
========================================= */

.steps {{
    margin-top: 25px;
    padding-top: 21px;

    border-top: 1px solid #edf1ed;

    text-align: left;
}}

.step {{
    display: flex;
    align-items: center;
    gap: 13px;

    margin-bottom: 17px;
}}

.step:last-child {{
    margin-bottom: 0;
}}

.step-number {{
    flex-shrink: 0;

    width: 37px;
    height: 37px;

    display: grid;
    place-items: center;

    border: 1px solid #dceade;
    border-radius: 11px;

    background: #f0f7f1;

    color: var(--primary);

    font-size: 10px;
    font-weight: 900;
}}

.step b {{
    display: block;

    margin-bottom: 4px;

    color: #314637;

    font-size: 12px;
    font-weight: 800;
}}

.step small {{
    display: block;

    color: #8b978d;

    font-size: 10px;
    line-height: 1.7;
}}

/* =========================================
   INFO STRIP
========================================= */

.bottom-info {{
    display: grid;

    grid-template-columns: repeat(3, 1fr);

    gap: 17px;

    margin-top: 25px;
}}

.info-item {{
    display: flex;
    align-items: center;
    gap: 12px;

    padding: 18px;

    border: 1px solid #e2ebe3;
    border-radius: 15px;

    background: rgba(255, 255, 255, 0.78);
}}

.info-icon {{
    flex-shrink: 0;

    width: 38px;
    height: 38px;

    display: grid;
    place-items: center;

    border-radius: 11px;

    background: var(--primary-soft);

    color: var(--primary);

    font-size: 17px;
    font-weight: 900;
}}

.info-item b {{
    display: block;

    margin-bottom: 4px;

    color: #314637;

    font-size: 11px;
    font-weight: 850;
}}

.info-item small {{
    display: block;

    color: #879387;

    font-size: 9px;
    line-height: 1.7;
}}

/* =========================================
   FOOTER
========================================= */

.footer {{
    padding: 25px 20px 30px;

    border-top: 1px solid rgba(24, 95, 53, 0.08);

    background: rgba(255, 255, 255, 0.72);

    text-align: center;
}}

.footer-brand {{
    margin-bottom: 9px;

    color: var(--primary);

    font-size: 16px;
    font-weight: 900;
}}

.footer p {{
    margin: 0;

    color: #839084;

    font-size: 10px;
    line-height: 1.9;
}}

/* =========================================
   ALERTS
========================================= */

.alert {{
    display: flex;
    align-items: center;
    gap: 11px;

    margin-bottom: 22px;
    padding: 15px 17px;

    border-radius: 12px;

    font-size: 12px;
    line-height: 1.7;
}}

.alert-icon {{
    flex-shrink: 0;

    display: grid;
    place-items: center;

    width: 25px;
    height: 25px;

    border-radius: 50%;

    font-weight: 900;
}}

.error {{
    border: 1px solid #f3cccc;

    background: #fff5f5;

    color: #9d3131;
}}

.error .alert-icon {{
    background: #fce1e1;
    color: #a52c2c;
}}

/* =========================================
   ANIMATIONS
========================================= */

@keyframes fadeUp {{
    from {{
        opacity: 0;
        transform: translateY(17px);
    }}

    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes resultAppear {{
    from {{
        opacity: 0;
        transform: translateY(10px) scale(0.99);
    }}

    to {{
        opacity: 1;
        transform: translateY(0) scale(1);
    }}
}}

@keyframes floatingCard {{
    0%, 100% {{
        transform: translateY(0);
    }}

    50% {{
        transform: translateY(-8px);
    }}
}}

@keyframes buildingFloat {{
    0%, 100% {{
        filter: brightness(1);
    }}

    50% {{
        filter: brightness(1.035);
    }}
}}

@keyframes sunPulse {{
    0%, 100% {{
        transform: scale(1);
        opacity: 0.8;
    }}

    50% {{
        transform: scale(1.09);
        opacity: 1;
    }}
}}

@keyframes checkPulse {{
    0%, 100% {{
        transform: scale(1);
    }}

    50% {{
        transform: scale(1.08);
    }}
}}

/* =========================================
   RESPONSIVE DESIGN
========================================= */

@media (max-width: 900px) {{

    main {{
        padding: 35px 20px 50px;
    }}

    .layout {{
        grid-template-columns:
            minmax(0, 1fr)
            minmax(0, 0.95fr);

        gap: 17px;
    }}

    .panel-header {{
        padding: 21px 20px 18px;
    }}

    .panel-body {{
        padding: 22px 20px;
    }}

    .result-panel .panel-body {{
        padding: 20px;
    }}

    .illustration {{
        height: 220px;
    }}

    .bottom-info {{
        gap: 10px;
    }}

    .info-item {{
        padding: 13px;
        gap: 9px;
    }}
}}

@media (max-width: 700px) {{

    .topbar {{
        padding: 15px 18px;
    }}

    .brand-name {{
        font-size: 20px;
    }}

    main {{
        padding: 30px 16px 40px;
    }}

    .hero {{
        margin-bottom: 28px;
    }}

    .hero h1 {{
        font-size: clamp(34px, 8vw, 45px);
        letter-spacing: -1.6px;
    }}

    .hero-description {{
        font-size: 13px;
    }}

    .layout {{
        grid-template-columns: minmax(0, 1fr);
        gap: 19px;
    }}

    .result-panel {{
        order: 2;
    }}

    .form-panel {{
        order: 1;
    }}

    .bottom-info {{
        grid-template-columns: 1fr;
        gap: 10px;
    }}

    .info-item {{
        padding: 15px;
    }}

    .bg-building.a {{
        left: -35px;
    }}

    .bg-building.c {{
        right: -40px;
    }}

    .decor-circle.one {{
        left: -260px;
    }}

    .decor-circle.two {{
        right: -350px;
    }}
}}

@media (max-width: 430px) {{

    .topbar {{
        padding: 13px;
    }}

    .brand {{
        gap: 8px;
    }}

    .brand-mark {{
        width: 36px;
        height: 36px;
    }}

    .brand-name {{
        font-size: 18px;
    }}

    .header-status {{
        gap: 6px;
        padding: 8px;
        font-size: 8px;
        letter-spacing: 0;
    }}

    .status-dot {{
        width: 6px;
        height: 6px;
    }}

    main {{
        padding: 25px 12px 35px;
    }}

    .hero h1 {{
        font-size: 34px;
    }}

    .hero-label {{
        font-size: 9px;
        letter-spacing: 0.8px;
    }}

    .panel {{
        border-radius: 18px;
    }}

    .panel-header {{
        padding: 18px 16px;
    }}

    .panel-body {{
        padding: 20px 16px;
    }}

    .result-panel .panel-body {{
        padding: 17px;
    }}

    .panel-title {{
        font-size: 17px;
    }}

    .fields {{
        grid-template-columns: minmax(0, 1fr);
        gap: 18px;
    }}

    .field.full {{
        grid-column: auto;
    }}

    .illustration {{
        height: 218px;
    }}

    .building-back {{
        left: 22px;
    }}

    .building-front {{
        right: 20px;
    }}

    .floating-price {{
        left: 9px;
        top: 28px;
    }}

    .price {{
        font-size: 29px;
    }}
}}

/* Respect accessibility preferences */

@media (prefers-reduced-motion: reduce) {{
    *,
    *::before,
    *::after {{
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }}
}}

</style>

</head>

<body>

<header class="topbar">

    <a class="brand" href="/" aria-label="BDProperties Home">

        <div class="brand-mark">

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
                aria-hidden="true"
            >
                <path d="M3 10.5L12 3l9 7.5"></path>
                <path d="M5.5 9.5V21h13V9.5"></path>
                <path d="M9 21v-6h6v6"></path>
                <path d="M9 11h.01M15 11h.01"></path>
            </svg>

        </div>

        <div class="brand-name">
            BD<span>Properties</span>
        </div>

    </a>

    <div class="header-right">

        <div class="header-status">
            <span class="status-dot"></span>
            PROPERTY ESTIMATOR
        </div>

    </div>

</header>


<div class="page-shell">

    <div class="background-decoration" aria-hidden="true">

        <div class="decor-circle one"></div>
        <div class="decor-circle two"></div>
        <div class="decor-circle three"></div>

        <div class="bg-building a">
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
        </div>

        <div class="bg-building b">
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
        </div>

        <div class="bg-building c">
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
        </div>

        <div class="bg-building d">
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
            <div class="window-row"><i></i><i></i></div>
        </div>

        <div class="floating-particle particle-one"></div>
        <div class="floating-particle particle-two"></div>
        <div class="floating-particle particle-three"></div>
        <div class="floating-particle particle-four"></div>

    </div>


    <main>

        <section class="hero">

            <div class="hero-label">

                <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    aria-hidden="true"
                >
                    <path d="M3 10l9-7 9 7"></path>
                    <path d="M5 9v12h14V9"></path>
                    <path d="M9 21v-7h6v7"></path>
                </svg>

                BANGLADESH REAL ESTATE

            </div>

            <h1>
                Your Property.<br>
                <span class="highlight">Your Price Estimate.</span>
            </h1>

            <p class="hero-description">
                Discover an estimated listing price for your
                property with BDProperties. Enter your
                property details and get a machine-learning
                based estimate in Bangladeshi Taka.
            </p>

        </section>


        {error_html}


        <div class="layout">


            <!-- LEFT: PROPERTY DETAILS FORM -->

            <section class="panel form-panel">

                <div class="panel-header">

                    <div class="panel-icon">

                        <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="1.8"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            aria-hidden="true"
                        >
                            <path d="M4 21V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v16"></path>
                            <path d="M2 21h20"></path>
                            <path d="M8 7h2M8 11h2M8 15h2"></path>
                            <path d="M14 7h.01M14 11h.01"></path>
                            <path d="M10 21v-3h4v3"></path>
                        </svg>

                    </div>

                    <div>

                        <h2 class="panel-title">
                            Property Details
                        </h2>

                        <p class="panel-subtitle">
                            Tell us about the property you want
                            to estimate.
                        </p>

                    </div>

                </div>


                <div class="panel-body">

                    <form method="post" action="/predict">

                        <div class="fields">


                            <div class="field full">

                                <label for="property_type">
                                    Property Type
                                    <span class="required">*</span>
                                </label>

                                <div class="input-container">

                                    <select
                                        id="property_type"
                                        name="property_type"
                                        required
                                    >
                                        {type_options(prop_type)}
                                    </select>

                                </div>

                            </div>


                            <div class="field full">

                                <label for="location">
                                    Property Location
                                    <span class="required">*</span>
                                </label>

                                <div class="location-container">

                                    <input
                                        id="location"
                                        name="location"
                                        class="with-icon"
                                        type="text"
                                        list="location-options"
                                        value="{html.escape(location, quote=True)}"
                                        placeholder="Search your property location"
                                        autocomplete="off"
                                        required
                                    >

                                    <div class="input-symbol">

                                        <svg
                                            viewBox="0 0 24 24"
                                            fill="none"
                                            stroke="currentColor"
                                            stroke-width="1.8"
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            aria-hidden="true"
                                        >
                                            <circle cx="11" cy="11" r="7"></circle>
                                            <path d="m16 16 4 4"></path>
                                        </svg>

                                    </div>

                                    <span class="search-mark">⌄</span>

                                    <datalist id="location-options">
                                        {location_options()}
                                    </datalist>

                                </div>

                                <div class="location-tip">
                                    <span>i</span>
                                    Search and select a location
                                    from the available suggestions.
                                </div>

                            </div>


                            <div class="field full">

                                <label for="area_sqft">
                                    Property Area
                                    <span class="required">*</span>
                                </label>

                                <div class="input-container">

                                    <input
                                        id="area_sqft"
                                        name="area_sqft"
                                        type="number"
                                        min="100"
                                        max="1000000"
                                        step="1"
                                        value="{html.escape(str(area), quote=True)}"
                                        placeholder="Enter area in square feet"
                                        required
                                    >

                                </div>

                                <p class="field-help">
                                    Example: 1200 square feet.
                                </p>

                            </div>


                            <div class="field">

                                <label for="bedrooms">
                                    Number of Bedrooms
                                </label>

                                <input
                                    id="bedrooms"
                                    name="bedrooms"
                                    type="number"
                                    min="0"
                                    max="30"
                                    step="1"
                                    value="{html.escape(str(beds), quote=True)}"
                                    placeholder="e.g. 3"
                                >

                            </div>


                            <div class="field">

                                <label for="bathrooms">
                                    Number of Bathrooms
                                </label>

                                <input
                                    id="bathrooms"
                                    name="bathrooms"
                                    type="number"
                                    min="0"
                                    max="30"
                                    step="1"
                                    value="{html.escape(str(baths), quote=True)}"
                                    placeholder="e.g. 2"
                                >

                            </div>


                        </div>


                        <button
                            class="predict-button"
                            type="submit"
                            {disabled}
                        >

                            <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                aria-hidden="true"
                            >
                                <path d="M12 3v3"></path>
                                <path d="M12 18v3"></path>
                                <path d="M3 12h3"></path>
                                <path d="M18 12h3"></path>
                                <path d="m5.6 5.6 2.1 2.1"></path>
                                <path d="m16.3 16.3 2.1 2.1"></path>
                                <path d="m18.4 5.6-2.1 2.1"></path>
                                <path d="m7.7 16.3-2.1 2.1"></path>
                                <circle cx="12" cy="12" r="4"></circle>
                            </svg>

                            Estimate Property Price

                            <span>→</span>

                        </button>


                        <div class="form-footnote">

                            <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                aria-hidden="true"
                            >
                                <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"></path>
                                <path d="m9 12 2 2 4-4"></path>
                            </svg>

                            Your estimate is generated from the
                            available property dataset.

                        </div>


                    </form>

                </div>

            </section>


            <!-- RIGHT: PRICE ESTIMATE -->

            <section class="panel result-panel">

                <div class="panel-header">

                    <div class="panel-icon">

                        <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="1.8"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            aria-hidden="true"
                        >
                            <rect x="3" y="5" width="18" height="14" rx="2"></rect>
                            <path d="M16 12h.01"></path>
                            <path d="M7 9h5"></path>
                            <path d="M7 15h3"></path>
                        </svg>

                    </div>

                    <div>

                        <h2 class="panel-title">
                            Property Valuation
                        </h2>

                        <p class="panel-subtitle">
                            Your estimated listing price appears here.
                        </p>

                    </div>

                </div>


                <div class="panel-body">

                    {result_html}

                </div>

            </section>


        </div>


        <!-- TRUST AND INFORMATION STRIP -->

        <section class="bottom-info">

            <div class="info-item">

                <div class="info-icon">⌕</div>

                <div>
                    <b>Location-Based</b>
                    <small>
                        Uses the selected dataset location.
                    </small>
                </div>

            </div>


            <div class="info-item">

                <div class="info-icon">↗</div>

                <div>
                    <b>Machine Learning</b>
                    <small>
                        Uses your property details for estimation.
                    </small>
                </div>

            </div>


            <div class="info-item">

                <div class="info-icon">৳</div>

                <div>
                    <b>Bangladeshi Taka</b>
                    <small>
                        View estimates in BDT and lakh.
                    </small>
                </div>

            </div>

        </section>


    </main>

</div>


<footer class="footer">

    <div class="footer-brand">
        BDProperties
    </div>

    <p>
        Property price estimation for Bangladesh.<br>
        Estimates are informational and are not
        guaranteed market or transaction prices.
    </p>

</footer>


</body>
</html>
"""