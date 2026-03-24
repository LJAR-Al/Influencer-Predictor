"""Score new creators: conversion likelihood + price/CPM ranges."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import QUANTILES, PROFITABILITY_THRESHOLD

MODELS_DIR = Path(__file__).parent.parent / "models"


def save_models(models):
    """Save all models to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    for name, pipeline in models.items():
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")


def load_models():
    """Load all models from disk."""
    models = {}
    for name in ["classifier"] + list(QUANTILES.keys()) + ["l2"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


def score_creator(X, expected_views, models=None):
    """
    Score a creator and return conversion probability + price/CPM ranges.

    Args:
        X: Feature dataframe (from build_features).
        expected_views: Expected view count for the campaign.
        models: Dict of models. Loaded from disk if None.

    Returns:
        Dict with conversion_probability and pricing DataFrame.
    """
    if models is None:
        models = load_models()

    clf = models["classifier"]
    conv_prob = clf.predict_proba(X)[:, 1]

    ev = np.atleast_1d(np.array(expected_views, dtype=float))

    # Collect raw predictions per quantile
    quantile_names = [name for name in QUANTILES if name in models]
    raw_revenues = {}
    for name in quantile_names:
        log_pred = models[name].predict(X)
        log_pred = np.where(conv_prob >= 0.5, log_pred, 0.0)
        raw_revenues[name] = np.expm1(log_pred)

    # Enforce monotonic ordering: conservative <= moderate <= aggressive
    for i in range(len(X)):
        vals = [raw_revenues[name][i] for name in quantile_names]
        vals_sorted = sorted(vals)
        for j, name in enumerate(quantile_names):
            raw_revenues[name][i] = vals_sorted[j]

    rows = []
    for name in quantile_names:
        revenue = raw_revenues[name]
        max_price = revenue / PROFITABILITY_THRESHOLD
        max_cpm = max_price / (ev / 1000)

        for i in range(len(X)):
            rows.append({
                "level": name,
                "predicted_iap_revenue": round(float(revenue[i]), 2),
                "max_price": round(float(max_price[i]), 2),
                "max_cpm": round(float(max_cpm[i]), 2),
                "expected_views": int(ev[i]) if len(ev) > 1 else int(ev[0]),
            })

    return {
        "conversion_probability": float(conv_prob[0]) if len(X) == 1 else conv_prob.tolist(),
        "pricing": pd.DataFrame(rows),
    }


def format_scorecard(X, expected_views, models=None, creator_name="Creator"):
    """Format a human-readable pricing scorecard."""
    result = score_creator(X, expected_views, models)
    conv_pct = result["conversion_probability"]
    df = result["pricing"]

    lines = [
        f"  {creator_name}",
        f"  Expected Reach: {expected_views:,} views",
        f"  Conversion Likelihood: {conv_pct:.1%}",
        f"  Profitability: IAP >= {PROFITABILITY_THRESHOLD:.0%} of price",
        f"  {'─'*60}",
        f"  {'Level':<15} {'Pred IAP Rev':>14} {'Max Price':>12} {'Max CPM':>10}",
        f"  {'─'*60}",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"  {row['level']:<15} "
            f"${row['predicted_iap_revenue']:>12,.2f} "
            f"${row['max_price']:>10,.2f} "
            f"${row['max_cpm']:>8,.2f}"
        )

    lines.append(f"  {'─'*60}")
    return "\n".join(lines)
