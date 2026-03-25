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

Returning creators are automatically detected and priced using their
actual historical conversion rate and APPU instead of benchmarks.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import (
    QUANTILES, PROFITABILITY_THRESHOLD, PRE_CAMPAIGN_NUMERIC, PRE_CAMPAIGN_CATEGORICAL,
    CLASSIFIER_THRESHOLD, DEFAULT_NEW_LEVEL, DEFAULT_REBOOKING_LEVEL,
)
from src.predict import load_models
from src.rebooking import build_creator_profiles, price_rebooking
from src.dynamic_benchmarks import compute_segmented_benchmarks, get_benchmark_for_creator
from src.data import load_raw, clean


def score_batch(input_path, output_path=None):
    clf, benchmarks = load_models()

    # Load training data for rebooking profiles
    df_training = clean(load_raw())
    profiles = build_creator_profiles(df_training)

    df = pd.read_csv(input_path)
    original = df.copy()

    # Normalize platform names
    if "posting_platform" in df.columns:
        df["posting_platform"] = df["posting_platform"].str.strip().str.title()

    # Fill missing optional columns
    for col in PRE_CAMPAIGN_NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in PRE_CAMPAIGN_CATEGORICAL:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown")

    # Creator history columns
    df["creator_prior_campaigns"] = 0
    df["creator_avg_iap"] = 0.0
    df["creator_avg_iap_per_1k_views"] = 0.0

    feature_cols = PRE_CAMPAIGN_NUMERIC + [
        "creator_prior_campaigns", "creator_avg_iap", "creator_avg_iap_per_1k_views",
    ] + PRE_CAMPAIGN_CATEGORICAL
    X = df[[c for c in feature_cols if c in df.columns]]

    # Classifier scores
    conv_prob = clf.predict_proba(X)[:, 1]
    original["conversion_likelihood"] = (conv_prob * 100).round(1)
    original["low_conversion_flag"] = conv_prob < CLASSIFIER_THRESHOLD

    ev = df["expected_views"].values.astype(float)
    name_col = "creator_name" if "creator_name" in df.columns else None

    # Initialize columns
    original["is_rebooking"] = False
    original["prior_campaigns"] = 0
    original["rebooking_conv_rate"] = np.nan
    original["rebooking_appu"] = np.nan
    original["rebooking_predicted_iap"] = np.nan
    original["rebooking_max_price"] = np.nan
    original["rebooking_max_cpm"] = np.nan
    original["hist_view_ratio"] = np.nan

    # Dynamic benchmark pricing (segmented by country + gender + reach + category)
    all_benchmarks = compute_segmented_benchmarks(df_training)
    original["benchmark_segment"] = ""
    original["benchmark_adjustments"] = ""

    for qname in QUANTILES:
        original[f"{qname}_max_cpm"] = 0.0
        original[f"{qname}_max_price"] = 0.0
        original[f"{qname}_min_iap_needed"] = 0.0

    for i in range(len(original)):
        country = df.loc[i, "demographics_main_country"] if "demographics_main_country" in df.columns else "Unknown"
        fem_pct = df.loc[i, "demographics_female_pct"] if "demographics_female_pct" in df.columns else 0
        category = df.loc[i, "youtube_category_name"] if "youtube_category_name" in df.columns else "Unknown"

        bm, segment, adjustments = get_benchmark_for_creator(
            country, fem_pct, ev[i], category, all_benchmarks,
        )
        original.loc[i, "benchmark_segment"] = segment
        original.loc[i, "benchmark_adjustments"] = adjustments

        # Always compute benchmark pricing regardless of conversion likelihood
        # Low conversion is a flag, not a reason to zero out pricing
        for qname in QUANTILES:
            cpm = bm[qname]
            max_price = cpm * (ev[i] / 1000)
            original.loc[i, f"{qname}_max_cpm"] = round(cpm, 2)
            original.loc[i, f"{qname}_max_price"] = round(max_price, 2)
            original.loc[i, f"{qname}_min_iap_needed"] = round(max_price * PROFITABILITY_THRESHOLD, 2)

    # Rebooking: blend actual data with benchmark based on confidence
    rebooking_count = 0
    for i in range(len(original)):
        if name_col is None:
            continue
        creator_name = original.loc[i, name_col]
        if pd.isna(creator_name):
            continue

        # Pass moderate benchmark price for blending
        # Use aggressive benchmark for rebooking blend (backtested: better ROI)
        benchmark_price = original.loc[i, f"{DEFAULT_REBOOKING_LEVEL}_max_price"]
        result = price_rebooking(creator_name, ev[i], profiles, benchmark_max_price=benchmark_price)
        if result is None:
            continue

        rebooking_count += 1
        original.loc[i, "is_rebooking"] = True
        original.loc[i, "prior_campaigns"] = result["prior_campaigns"]
        original.loc[i, "rebooking_conv_rate"] = result["conversion_rate"]
        original.loc[i, "rebooking_appu"] = result["appu"]
        original.loc[i, "rebooking_predicted_iap"] = result["predicted_iap"]
        original.loc[i, "rebooking_max_price"] = result["blended_max_price"]
        original.loc[i, "rebooking_max_cpm"] = result["blended_max_cpm"]
        original.loc[i, "hist_view_ratio"] = result["avg_view_ratio"]

    # Asking price comparison (vs moderate for new, vs rebooking for returning)
    has_asking = "asking_price" in original.columns
    if has_asking:
        asking = pd.to_numeric(original["asking_price"], errors="coerce")
        # Use rebooking blended price if available, otherwise moderate benchmark
        best_price = np.where(
            original["is_rebooking"],
            original["rebooking_max_price"],
            original[f"{DEFAULT_NEW_LEVEL}_max_price"],
        )
        original["best_max_price"] = np.round(best_price, 2)
        original["asking_vs_best"] = (asking - best_price).round(2)
        original["asking_vs_best_pct"] = np.where(
            asking > 0,
            ((original["asking_vs_best"] / asking) * 100).round(1),
            0.0,
        )

    # Output
    if output_path is None:
        output_path = input_path.replace(".csv", "_scored.csv")
    original.to_csv(output_path, index=False)
    print(f"Scored {len(original)} creators → {output_path}")
    print(f"  Rebookings (actual data): {rebooking_count}")
    print(f"  New creators (dynamic benchmark): {len(original) - rebooking_count}")
    print(f"  Segments used: {original['benchmark_segment'].value_counts().to_dict()}")

    # Print summary
    print(f"\n{'─'*110}")

    for i, row in original.iterrows():
        label = row[name_col] if name_col else f"Creator #{i+1}"
        conv = row["conversion_likelihood"]
        reach = int(row["expected_views"])
        is_rb = row["is_rebooking"]

        adj = row.get("benchmark_adjustments", "")
        low_conv = row.get("low_conversion_flag", False)
        tag = f"★ REBOOKING ({int(row['prior_campaigns'])} prior)" if is_rb else f"NEW [{adj}]"
        flag = "  ⚠ LOW CONVERSION SIGNAL" if low_conv else ""
        print(f"\n  {label}  |  {tag}  |  Conversion: {conv}%  |  Reach: {reach:,}{flag}")

        if is_rb:
            vr = row["hist_view_ratio"]
            vr_label = f"{vr:.0%} of expected" if pd.notna(vr) else "N/A"
            print(f"  Conv rate: {row['rebooking_conv_rate']:.4%}  |  APPU: ${row['rebooking_appu']:.2f}  |  Hist view delivery: {vr_label}")
            print(f"  Blended max: ${row['rebooking_max_price']:,.0f} (CPM ${row['rebooking_max_cpm']:.2f})  |  Pred IAP: ${row['rebooking_predicted_iap']:,.0f}")

        if has_asking and pd.notna(row.get("asking_price")) and row["asking_price"] > 0:
            ask = row["asking_price"]
            best = row["best_max_price"]
            diff = row["asking_vs_best"]
            diff_pct = row["asking_vs_best_pct"]
            print(f"  Asking: ${ask:,.0f}  |  Best max: ${best:,.0f}  |  Diff: ${diff:+,.0f} ({diff_pct:+.0f}%)")

        if not is_rb:
            print(f"  {'Level':<15} {'Max CPM':>10} {'Max Price':>12} {'Min IAP':>12}")
            print(f"  {'─'*52}")
            for lv in QUANTILES:
                print(f"  {lv:<15} ${row[f'{lv}_max_cpm']:>8,.2f} ${row[f'{lv}_max_price']:>10,.0f} ${row[f'{lv}_min_iap_needed']:>10,.0f}")

    return original


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    score_batch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
