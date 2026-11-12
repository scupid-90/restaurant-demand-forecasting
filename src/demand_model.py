"""
Restaurant Sales Analysis — Demand Forecasting & Waste-Minimizing Prep Recommendations
=======================================================================================

What this script does
----------------------
1. Loads restaurant_sales_data.csv and engineers time/context features.
2. Trains a Random Forest model to predict `quantity_sold` for a given
   (restaurant type, menu item, meal type, weekday, weather, promotion,
   special event) combination.
3. Uses the model's prediction error (per item) to set a safety-stock
   buffer, and turns that into a concrete prep recommendation that
   balances two costs:
     - Overproduction cost  = ingredients wasted on unsold food
     - Underproduction cost = lost profit margin on demand you could
       not fill
   This is the classic "newsvendor" trade-off, applied per menu item.
4. Exposes `recommend_prep(...)` so this logic can be reused by a CLI,
   an API, or (as here) exported to JSON for the web dashboard.

Run directly to train, evaluate, and print sample recommendations:
    python3 demand_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "restaurant_sales_data.csv"
CAT_FEATURES = [
    "restaurant_type",
    "menu_item_name",
    "meal_type",
    "weekday",
    "weather_condition",
]
BOOL_FEATURES = ["has_promotion", "special_event"]
TARGET = "quantity_sold"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["weekday"] = df["date"].dt.day_name()
    df["revenue"] = df["actual_selling_price"] * df["quantity_sold"]
    df["margin_per_unit"] = df["actual_selling_price"] - df["typical_ingredient_cost"]
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="passthrough",  # passes through the two boolean columns
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def train(df: pd.DataFrame):
    features = CAT_FEATURES + BOOL_FEATURES
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    baseline = y_train.mean()
    baseline_mae = mean_absolute_error(y_test, [baseline] * len(y_test))
    return pipe, {
        "mae": round(mae, 1),
        "rmse": round(rmse, 1),
        "naive_baseline_mae": round(baseline_mae, 1),
        "improvement_vs_naive_pct": round(100 * (1 - mae / baseline_mae), 1),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def per_item_residual_std(pipe: Pipeline, df: pd.DataFrame) -> dict:
    """Model error varies by item (a drink is easier to predict than a
    fine-dining dish). Compute each item's residual std so the safety
    buffer below is sized per item, not with one global number."""
    features = CAT_FEATURES + BOOL_FEATURES
    preds = pipe.predict(df[features])
    resid = df[TARGET].values - preds
    out = {}
    for item in df["menu_item_name"].unique():
        mask = df["menu_item_name"] == item
        out[item] = float(np.std(resid[mask])) if mask.sum() > 1 else float(np.std(resid))
    return out


def recommend_prep(
    pipe: Pipeline,
    item_econ: dict,
    residual_std: dict,
    restaurant_type: str,
    menu_item_name: str,
    meal_type: str,
    weekday: str,
    weather_condition: str,
    has_promotion: bool,
    special_event: bool,
) -> dict:
    """Newsvendor-style prep recommendation for one item / one service.

    Critical ratio = underage_cost / (underage_cost + overage_cost)
      - overage_cost  (cost of one unit made but not sold) = ingredient cost
      - underage_cost (profit lost on one unit of unmet demand) = margin

    We prep at the critical-ratio quantile of the predicted demand
    distribution (assumed ~Normal(predicted_mean, item_residual_std)),
    which is the quantity that minimizes expected waste + lost-sales cost.
    """
    row = pd.DataFrame([{
        "restaurant_type": restaurant_type,
        "menu_item_name": menu_item_name,
        "meal_type": meal_type,
        "weekday": weekday,
        "weather_condition": weather_condition,
        "has_promotion": has_promotion,
        "special_event": special_event,
    }])
    predicted = float(pipe.predict(row)[0])

    econ = item_econ[menu_item_name]
    overage_cost = econ["ingredient_cost"]
    underage_cost = econ["margin"]
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    critical_ratio = min(max(critical_ratio, 0.5), 0.97)  # keep it sane

    std = residual_std.get(menu_item_name, predicted * 0.15)
    # z-score for the critical ratio (Normal quantile), via a light
    # approximation so this file has no scipy dependency
    z = _norm_ppf(critical_ratio)
    recommended_prep = max(0, predicted + z * std)

    naive_prep = predicted  # what you'd prep if you just used the raw forecast
    expected_waste_units = max(0, recommended_prep - predicted)

    return {
        "predicted_demand": round(predicted, 1),
        "recommended_prep": round(recommended_prep, 1),
        "naive_prep": round(naive_prep, 1),
        "extra_units_vs_naive": round(recommended_prep - naive_prep, 1),
        "critical_ratio": round(critical_ratio, 3),
        "expected_waste_units": round(expected_waste_units, 1),
        "unit_ingredient_cost": round(overage_cost, 2),
        "unit_margin": round(underage_cost, 2),
    }


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the standard normal inverse CDF."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = (-2 * np.log(1 - p)) ** 0.5
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def item_economics(df: pd.DataFrame) -> dict:
    g = df.groupby("menu_item_name").agg(
        ingredient_cost=("typical_ingredient_cost", "mean"),
        selling_price=("actual_selling_price", "mean"),
    )
    g["margin"] = g["selling_price"] - g["ingredient_cost"]
    return g[["ingredient_cost", "margin"]].to_dict(orient="index")


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows | {df['menu_item_name'].nunique()} menu items | "
          f"{df['restaurant_id'].nunique()} restaurants | "
          f"{df['date'].min().date()} to {df['date'].max().date()}\n")

    pipe, metrics = train(df)
    print("Model performance (held-out 20% test set):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    econ = item_economics(df)
    resid_std = per_item_residual_std(pipe, df)

    print("Sample prep recommendations:\n")
    samples = [
        dict(restaurant_type="Food Stall", menu_item_name="Teh Tarik", meal_type="Breakfast",
             weekday="Saturday", weather_condition="Sunny", has_promotion=False, special_event=False),
        dict(restaurant_type="Casual Dining", menu_item_name="Chicken Chop", meal_type="Dinner",
             weekday="Friday", weather_condition="Rainy", has_promotion=True, special_event=False),
        dict(restaurant_type="Fine Dining", menu_item_name="Beef Rendang", meal_type="Dinner",
             weekday="Sunday", weather_condition="Cloudy", has_promotion=False, special_event=True),
    ]
    for s in samples:
        rec = recommend_prep(pipe, econ, resid_std, **s)
        print(f"  {s['menu_item_name']} ({s['weekday']}, {s['meal_type']}, {s['weather_condition']}"
              f"{', promo' if s['has_promotion'] else ''}{', event' if s['special_event'] else ''}):")
        print(f"    predicted demand   : {rec['predicted_demand']}")
        print(f"    recommended prep   : {rec['recommended_prep']}  "
              f"(+{rec['extra_units_vs_naive']} vs. naive forecast, "
              f"critical ratio {rec['critical_ratio']})")
        print()

    # Export everything the dashboard needs as static JSON (no server needed)
    out_dir = Path("/home/claude/analysis")
    export = {
        "metrics": metrics,
        "item_economics": econ,
        "residual_std": resid_std,
        "items": sorted(df["menu_item_name"].unique().tolist()),
        "restaurant_types": sorted(df["restaurant_type"].unique().tolist()),
        "meal_types": sorted(df["meal_type"].unique().tolist()),
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "weather_conditions": sorted(df["weather_condition"].unique().tolist()),
    }
    with open(out_dir / "model_export.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported model summary to {out_dir / 'model_export.json'}")
