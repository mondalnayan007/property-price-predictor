# BanglaProperties Property Price Predictor

A standalone FastAPI website that loads the existing CatBoost apartment-price model and serves a responsive input form at `/`. It can be hosted separately from the MERN website; use the deployed website URL for the existing “Property Price Predict” button.

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. API docs are available at `/docs`; health check at `/health`.

## Deploy to Render

1. Create a GitHub repository and upload all files in this folder (including the `.cbm` model and `locations.json`).
2. On Render, choose **New + → Blueprint** and connect the repository, or create a **Web Service** from it.
3. If using a regular Web Service, choose Python runtime and set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Wait until deployment succeeds, then open the Render-provided `https://....onrender.com` URL. The form should load there.
5. In BanglaProperties.com, set the existing button's destination to that deployed URL, opening it in the same or a new tab as you prefer.

## Model limitations

- The bundled model was trained on apartment listing/asking prices from the supplied dataset, not verified completed-sale transactions.
- This interface is for apartments in the model's Dhaka-area dataset only. It is not a formal appraisal, and estimates may be inaccurate.
- The model training/evaluation previously showed preliminary MAE of about BDT 1.72 million on a location-grouped holdout set. Do not advertise the estimate as guaranteed or transaction-verified.
