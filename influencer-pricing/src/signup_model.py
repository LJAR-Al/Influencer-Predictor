"""V2 pricing model: signup rate prediction + APPU-weighted IAP projection."""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import (
    PLAYBOOK_FEATURES, SIGNUP_MODEL_PARAMS, PROFITABILITY_THRESHOLD,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    DEMO_AGE_COLS, DEMO_GENDER_COLS, DEMO_TIER_COLS,
    TIER_1_COUNTRIES, TIER_2_COUNTRIES,
)

MODELS_DIR = Path(__file__).parent.parent / "models"


# ── APPU Lookup ──

def build_appu_lookup(purchase_demographics_path):
    """
    Build APPU lookup tables from platform purchase demographics data.

    Returns:
        ga_appu: dict of (gender, age_group) -> APPU (global, all geos)
        tier_appu: dict of tier (1, 2, 3) -> average APPU for that tier
        global_appu: float, overall average APPU
    """
    df = pd.read_csv(purchase_demographics_path)
    for c in ["total_user_purchases", "num_user"]:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float)

    # Gender × Age APPU (aggregated across all geos)
    ga = df.groupby(["gender", "age_group"]).agg(
        {"num_user": "sum", "total_user_purchases": "sum"}
    ).reset_index()
    ga["appu"] = ga["total_user_purchases"] / ga["num_user"]
    ga_appu = dict(zip(zip(ga["gender"], ga["age_group"]), ga["appu"]))

    # Global APPU
    global_appu = float(df["total_user_purchases"].sum() / df["num_user"].sum())

    # Per-tier APPU
    geo = df.groupby("user_country").agg(
        {"num_user": "sum", "total_user_purchases": "sum"}
    ).reset_index()

    def _tier_appu(countries):
        sub = geo[geo["user_country"].isin(countries)]
        return float(sub["total_user_purchases"].sum() / sub["num_user"].sum()) if sub["num_user"].sum() > 0 else global_appu

    tier_appu = {
        1: _tier_appu(TIER_1_COUNTRIES),
        2: _tier_appu(TIER_2_COUNTRIES),
        3: _tier_appu(
            geo[~geo["user_country"].isin(TIER_1_COUNTRIES + TIER_2_COUNTRIES)]["user_country"].tolist()
        ),
    }

    return ga_appu, tier_appu, global_appu


def compute_weighted_appu(row, ga_appu, tier_appu, global_appu):
    """
    Compute weighted APPU for a single creator based on their audience demographics.

    row must contain: calc_pct_gender_female, calc_pct_gender_male,
        calc_pct_age_16_20, calc_pct_age_21_29, calc_pct_age_30_49, calc_pct_age_50plus,
        calc_pct_tier_1, calc_pct_tier_2, calc_pct_tier_3, calc_pct_tier_other
    """
    # Gender distribution (normalize to sum to 1)
    gender_pcts = {}
    for col, label in DEMO_GENDER_COLS.items():
        gender_pcts[label] = float(row.get(col) or 0)
    g_total = sum(gender_pcts.values())
    if g_total > 0:
        gender_pcts = {k: v / g_total for k, v in gender_pcts.items()}
    else:
        gender_pcts = {"FEMALE": 0.5, "MALE": 0.5}

    # Age distribution (normalize to sum to 1)
    age_pcts = {}
    for col, label in DEMO_AGE_COLS.items():
        age_pcts[label] = float(row.get(col) or 0)
    a_total = sum(age_pcts.values())
    if a_total > 0:
        age_pcts = {k: v / a_total for k, v in age_pcts.items()}
    else:
        age_pcts = {v: 0.25 for v in DEMO_AGE_COLS.values()}

    # Base APPU: weighted sum across gender × age
    base_appu = sum(
        gender_pcts[g] * age_pcts[a] * ga_appu.get((g, a), global_appu)
        for g in gender_pcts
        for a in age_pcts
    )

    # Geo tier adjustment
    t1 = float(row.get("calc_pct_tier_1") or 0)
    t2 = float(row.get("calc_pct_tier_2") or 0)
    t3 = float(row.get("calc_pct_tier_3") or 0)
    t_other = float(row.get("calc_pct_tier_other") or 0)
    t_total = t1 + t2 + t3 + t_other

    if t_total > 0:
        geo_weighted = (
            t1 / t_total * tier_appu[1]
            + t2 / t_total * tier_appu[2]
            + (t3 + t_other) / t_total * tier_appu[3]
        )
        geo_mult = geo_weighted / global_appu
    else:
        geo_mult = 1.0

    return base_appu * geo_mult


# ── Signup Rate Model ──

def _build_signup_pipeline():
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"
        ), PLAYBOOK_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(**SIGNUP_MODEL_PARAMS)),
    ])


def train_signup_model(X, y_log_signup_rate):
    """Train signup rate regression on playbook features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log_signup_rate, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    pipeline = _build_signup_pipeline()

    # Cross-validate
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=CV_FOLDS, scoring="r2")
    cv_mae = -cross_val_score(pipeline, X_train, y_train, cv=CV_FOLDS, scoring="neg_mean_absolute_error")

    # Fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_r2": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_mae": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
    }

    return pipeline, metrics


def predict_signup_rate(X_playbook, model, training_rates=None):
    """
    Predict signup rate from playbook features. Returns array of rates.

    If training_rates is provided (array of actual signup rates from training),
    predictions beyond P95 of training data are flagged as outliers.
    """
    log_sr = model.predict(X_playbook)
    return np.expm1(log_sr)


def flag_outlier_predictions(predicted_sr, training_signup_rates):
    """
    Flag predictions that are beyond the P95 of training data.
    Returns array of booleans (True = outlier / high prediction).
    """
    p95 = np.percentile(training_signup_rates, 95)
    return predicted_sr > p95, p95


def get_signup_feature_importance(pipeline):
    model = pipeline.named_steps["model"]
    encoder = pipeline.named_steps["preprocessor"].transformers_[0][1]
    feat_names = list(encoder.get_feature_names_out(PLAYBOOK_FEATURES))
    return pd.DataFrame({
        "feature": feat_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


# ── Pricing ──

def price_creator(expected_views, predicted_signup_rate, weighted_appu):
    """
    Compute max price and breakeven CPM from model outputs.

    projected_iap = expected_views × predicted_signup_rate × weighted_appu
    max_price = projected_iap / PROFITABILITY_THRESHOLD
    breakeven_cpm = max_price / (expected_views / 1000)
    """
    projected_iap = expected_views * predicted_signup_rate * weighted_appu
    max_price = projected_iap / PROFITABILITY_THRESHOLD
    breakeven_cpm = max_price / (expected_views / 1000) if expected_views > 0 else 0.0
    return {
        "projected_iap": float(projected_iap),
        "max_price": float(max_price),
        "breakeven_cpm": float(breakeven_cpm),
    }


def format_scorecard_v2(
    creator_name, expected_views,
    predicted_signup_rate, weighted_appu, pitched_cpm=None,
):
    """Human-readable pricing scorecard for the V2 model."""
    pricing = price_creator(expected_views, predicted_signup_rate, weighted_appu)

    lines = [
        f"  {creator_name}",
        f"  Expected Reach: {expected_views:,} views",
        f"  Predicted Signup Rate: {predicted_signup_rate:.4%}",
        f"  Weighted APPU: ${weighted_appu:.2f}",
        f"  {'─' * 55}",
        f"  Projected Signups:  {expected_views * predicted_signup_rate:,.0f}",
        f"  Projected IAP:      ${pricing['projected_iap']:>10,.2f}",
        f"  Max Price (10% rule): ${pricing['max_price']:>10,.2f}",
        f"  Breakeven CPM:      ${pricing['breakeven_cpm']:>10,.2f}",
    ]

    if pitched_cpm is not None:
        pitched_total = pitched_cpm * (expected_views / 1000)
        verdict = "BOOK" if pricing["max_price"] >= pitched_total else "PASS"
        margin = pricing["projected_iap"] / pitched_total - 1 if pitched_total > 0 else 0
        lines.extend([
            f"  {'─' * 55}",
            f"  Pitched CPM:        ${pitched_cpm:>10,.2f}",
            f"  Pitched Total:      ${pitched_total:>10,.2f}",
            f"  Projected Margin:   {margin:>+10.1%}",
            f"  Verdict:            {verdict}",
        ])

    lines.append(f"  {'─' * 55}")
    return "\n".join(lines)


# ── Save / Load ──

def save_v2_models(signup_pipeline, ga_appu, tier_appu, global_appu):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(signup_pipeline, MODELS_DIR / "signup_model.joblib")
    appu_data = {
        "ga_appu": {f"{g}|{a}": v for (g, a), v in ga_appu.items()},
        "tier_appu": {str(k): v for k, v in tier_appu.items()},
        "global_appu": global_appu,
    }
    with open(MODELS_DIR / "appu_lookup.json", "w") as f:
        json.dump(appu_data, f, indent=2)


def load_v2_models():
    signup_pipeline = joblib.load(MODELS_DIR / "signup_model.joblib")
    with open(MODELS_DIR / "appu_lookup.json") as f:
        appu_data = json.load(f)
    ga_appu = {tuple(k.split("|")): v for k, v in appu_data["ga_appu"].items()}
    tier_appu = {int(k): v for k, v in appu_data["tier_appu"].items()}
    global_appu = appu_data["global_appu"]
    return signup_pipeline, ga_appu, tier_appu, global_appu
