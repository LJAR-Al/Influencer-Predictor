"""
Score a batch of creators from CSV and output pricing ranges.

Usage:
    python3 score_creators.py data/test_creators.csv

Input CSV columns (required):
    - expected_views          Average view count / reach
    - demographics_female_pct % female audience
    - demographics_male_pct   % male audience
    - demographics_other_pct  % other audience

Input CSV columns (optional but recommended):
    - expected_cpm             Pre-campaign CPM estimate
    - posting_platform         Youtube / Instagram / Tiktok
    - youtube_category_name    Entertainment, Gaming, Education, etc.
    - demographics_main_country  US, UK, DE, FR, etc.
    - creator_name             For display only
    - asking_price             Creator's initial price pitch

Output: same CSV with added columns for each pricing level.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import QUANTILES, PROFITABILITY_THRESHOLD, PRE_CAMPAIGN_NUMERIC, PRE_CAMPAIGN_CATEGORICAL
from src.predict import load_models


def score_batch(input_path, output_path=None):
    models = load_models()
    clf = models["classifier"]

    df = pd.read_csv(input_path)
    original = df.copy()

    # Normalize platform names
    if "posting_platform" in df.columns:
        df["posting_platform"] = df["posting_platform"].str.strip().str.title()

    # Fill missing optional columns with defaults
    for col in PRE_CAMPAIGN_NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in PRE_CAMPAIGN_CATEGORICAL:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown")

    # Add empty creator history columns (new creators have no history)
    df["creator_prior_campaigns"] = 0
    df["creator_avg_revenue"] = 0.0
    df["creator_avg_rev_per_1k_views"] = 0.0

    # Build feature matrix in the exact order the model expects
    feature_cols = PRE_CAMPAIGN_NUMERIC + [
        "creator_prior_campaigns", "creator_avg_revenue", "creator_avg_rev_per_1k_views",
    ] + PRE_CAMPAIGN_CATEGORICAL
    X = df[[c for c in feature_cols if c in df.columns]]

    # Score
    conv_prob = clf.predict_proba(X)[:, 1]
    original["conversion_likelihood"] = (conv_prob * 100).round(1)

    ev = df["expected_views"].values.astype(float)

    for name, q in QUANTILES.items():
        if name not in models:
            continue
        log_pred = models[name].predict(X)
        log_pred = np.where(conv_prob >= 0.5, log_pred, 0.0)
        revenue = np.expm1(log_pred)
        max_price = revenue / PROFITABILITY_THRESHOLD
        max_cpm = np.where(ev > 0, max_price / (ev / 1000), 0.0)

        original[f"{name}_predicted_iap"] = revenue.round(2)
        original[f"{name}_max_price"] = max_price.round(2)
        original[f"{name}_max_cpm"] = max_cpm.round(2)

    # Enforce monotonic ordering per row
    for i in range(len(original)):
        levels = list(QUANTILES.keys())
        for col_type in ["predicted_iap", "max_price", "max_cpm"]:
            vals = [original.loc[i, f"{lv}_{col_type}"] for lv in levels]
            vals_sorted = sorted(vals)
            for j, lv in enumerate(levels):
                original.loc[i, f"{lv}_{col_type}"] = vals_sorted[j]

    # Add asking price comparison (moderate = midpoint)
    has_asking = "asking_price" in original.columns
    if has_asking:
        asking = pd.to_numeric(original["asking_price"], errors="coerce")
        original["asking_vs_moderate"] = (asking - original["moderate_max_price"]).round(2)
        original["asking_vs_moderate_pct"] = ((original["asking_vs_moderate"] / asking) * 100).round(1)

    # Output
    if output_path is None:
        output_path = input_path.replace(".csv", "_scored.csv")
    original.to_csv(output_path, index=False)
    print(f"Scored {len(original)} creators → {output_path}")

    # Print summary
    print(f"\n{'─'*100}")
    name_col = "creator_name" if "creator_name" in original.columns else None

    for i, row in original.iterrows():
        label = row[name_col] if name_col else f"Creator #{i+1}"
        conv = row["conversion_likelihood"]
        reach = int(row["expected_views"])

        print(f"\n  {label}  |  Conversion: {conv}%  |  Reach: {reach:,}")

        if has_asking and pd.notna(row.get("asking_price")):
            ask = row["asking_price"]
            mod_price = row["moderate_max_price"]
            diff = row["asking_vs_moderate"]
            diff_pct = row["asking_vs_moderate_pct"]
            print(f"  Asking: ${ask:,.0f}  |  Moderate max: ${mod_price:,.0f}  |  Diff: ${diff:+,.0f} ({diff_pct:+.0f}%)")

        print(f"  {'Level':<15} {'Pred IAP':>10} {'Max Price':>12} {'Max CPM':>10}")
        print(f"  {'─'*50}")
        for lv in QUANTILES:
            print(f"  {lv:<15} ${row[f'{lv}_predicted_iap']:>8,.0f} ${row[f'{lv}_max_price']:>10,.0f} ${row[f'{lv}_max_cpm']:>8,.2f}")

    return original


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    score_batch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
