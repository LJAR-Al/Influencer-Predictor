"""Score new creators: conversion likelihood + benchmark-based price/CPM ranges."""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from src.config import QUANTILES, PROFITABILITY_THRESHOLD, CLASSIFIER_THRESHOLD

MODELS_DIR = Path(__file__).parent.parent / "models"


def save_models(clf, benchmarks):
    """Save classifier and CPM benchmarks."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "classifier.joblib")
    with open(MODELS_DIR / "benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)


def load_models():
    """Load classifier and benchmarks."""
    clf = joblib.load(MODELS_DIR / "classifier.joblib")
    with open(MODELS_DIR / "benchmarks.json") as f:
        benchmarks = json.load(f)
    return clf, benchmarks


def score_creator(X, expected_views, clf=None, benchmarks=None):
    """
    Score creators using classifier + profitable CPM benchmarks.

    For predicted converters, max price = benchmark CPM * (views / 1000).
    The benchmark CPMs come from historically profitable campaigns
    (where IAP >= 10% of cost).
    """
    if clf is None or benchmarks is None:
        clf, benchmarks = load_models()

    conv_prob = clf.predict_proba(X)[:, 1]
    ev = np.atleast_1d(np.array(expected_views, dtype=float))

    rows = []
    for name, q in QUANTILES.items():
        cpm = benchmarks[name]
        max_price = np.where(conv_prob >= CLASSIFIER_THRESHOLD, cpm * (ev / 1000), 0.0)
        predicted_iap = max_price * PROFITABILITY_THRESHOLD

        for i in range(len(X)):
            rows.append({
                "level": name,
                "benchmark_cpm": round(cpm, 2),
                "max_price": round(float(max_price[i]), 2),
                "max_cpm": round(cpm if conv_prob[i] >= 0.5 else 0.0, 2),
                "expected_iap_at_max_price": round(float(predicted_iap[i]), 2),
                "expected_views": int(ev[i]) if len(ev) > 1 else int(ev[0]),
            })

    return {
        "conversion_probability": float(conv_prob[0]) if len(X) == 1 else conv_prob.tolist(),
        "pricing": pd.DataFrame(rows),
    }


def format_scorecard(X, expected_views, clf=None, benchmarks=None, creator_name="Creator"):
    """Format a human-readable pricing scorecard."""
    result = score_creator(X, expected_views, clf, benchmarks)
    conv_pct = result["conversion_probability"]
    df = result["pricing"]

    lines = [
        f"  {creator_name}",
        f"  Expected Reach: {expected_views:,} views",
        f"  Conversion Likelihood: {conv_pct:.1%}",
        f"  Rule: IAP >= {PROFITABILITY_THRESHOLD:.0%} of price paid",
        f"  {'─'*65}",
        f"  {'Level':<15} {'Max CPM':>10} {'Max Price':>12} {'Min IAP needed':>15}",
        f"  {'─'*65}",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"  {row['level']:<15} "
            f"${row['max_cpm']:>8,.2f} "
            f"${row['max_price']:>10,.2f} "
            f"${row['expected_iap_at_max_price']:>13,.2f}"
        )

    lines.append(f"  {'─'*65}")
    return "\n".join(lines)
