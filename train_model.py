"""
Train a CatBoost property-price model from bproperty_data_cleaned.csv.

Run:
    python train_model.py --data bproperty_data_cleaned.csv

Outputs:
    property_price_model.cbm
    model_metadata.json
    model_metrics.json

The target is listing/asking price, not verified transaction price.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "Type", "Location", "No. Beds", "No. Baths", "Area",
    "Latitude", "Longitude", "Region", "Sub-region",
]
CAT_FEATURES = ["Type", "Location", "Region", "Sub-region"]
TARGET = "Price"
MISSING_CATEGORY = "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="bproperty_data_cleaned.csv",
                        help="Path to cleaned Bproperty CSV dataset")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Put bproperty_data_cleaned.csv "
            "in this folder or pass --data PATH."
        )

    df = pd.read_csv(data_path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].copy()

    numeric_features = ["No. Beds", "No. Baths", "Area", "Latitude", "Longitude"]
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Area"].notna() & (df["Area"] > 0)].copy()

    for col in CAT_FEATURES:
        df[col] = df[col].fillna(MISSING_CATEGORY).astype(str).str.strip()
        df.loc[df[col] == "", col] = MISSING_CATEGORY

    # Keep missing bedroom/bathroom values as NaN; CatBoost supports numeric NaNs.
    # Log target reduces the influence of very high-priced listings.
    X = df[FEATURES]
    y = np.log1p(df[TARGET].astype(float))

    # Hold out whole locations so evaluation is less optimistic than a random row split.
    groups = df["Location"].astype(str)
    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(df))
        cut = max(1, int(len(df) * 0.8))
        train_idx, test_idx = indices[:cut], indices[cut:]

    model = CatBoostRegressor(
        iterations=args.iterations,
        depth=8,
        learning_rate=0.04,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.seed,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        cat_features=CAT_FEATURES,
        eval_set=(X.iloc[test_idx], y.iloc[test_idx]) if len(test_idx) else None,
        early_stopping_rounds=100 if len(test_idx) else None,
        verbose=100,
    )

    output_dir = Path(__file__).resolve().parent
    model.save_model(str(output_dir / "property_price_model.cbm"))

    if len(test_idx):
        pred = np.maximum(0, np.expm1(model.predict(X.iloc[test_idx])))
        actual = df[TARGET].iloc[test_idx].to_numpy(dtype=float)
        mae = float(mean_absolute_error(actual, pred))
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        r2 = float(r2_score(actual, pred)) if len(actual) > 1 else None
        median_ae = float(np.median(np.abs(actual - pred)))
        metrics = {
            "rows_used": int(len(df)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "mae_bdt": mae,
            "median_absolute_error_bdt": median_ae,
            "rmse_bdt": rmse,
            "r2": r2,
            "target_note": "Listing/asking price; not verified transaction price.",
            "split_note": "Group split by exact Location; unseen-location performance can vary.",
        }
        (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\nEvaluation on held-out locations:")
        print(json.dumps(metrics, indent=2))

    # Metadata lets app.py display choices and fill geographic fields consistently.
    location_rows = (
        df[["Location", "Region", "Sub-region", "Latitude", "Longitude"]]
        .dropna(subset=["Location"])
        .drop_duplicates(subset=["Location"])
        .sort_values("Location")
    )
    type_values = sorted(df["Type"].dropna().astype(str).unique().tolist())
    metadata = {
        "features": FEATURES,
        "categorical_features": CAT_FEATURES,
        "types": type_values,
        "locations": location_rows.where(pd.notna(location_rows), None).to_dict(orient="records"),
        "rows_used": int(len(df)),
        "target_note": "Estimated listing/asking price in BDT; not a verified transaction price.",
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSaved property_price_model.cbm and model_metadata.json")


if __name__ == "__main__":
    main()
