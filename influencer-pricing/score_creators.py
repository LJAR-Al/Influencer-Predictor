"""
Score a batch of creators from CSV and output pricing ranges.

Usage:
    python3 score_creators.py data/test_creators_v2.csv

Three-layer evaluation:
    Layer 1 — Pre-screening: skip profiles with low-IAP audiences,
              suspicious view/sub ratios, or unreliable view delivery.
    Layer 2 — Conversion prediction: estimate signups from views using
              the V2 signup rate model (playbook features) or V1 classifier.
    Layer 3 — Revenue projection: multiply predicted signups by weighted APPU
              (from Freecash demographics) to get projected IAP and max price.

Input CSV columns (required):
    - expected_views          Average view count / reach
    - demographics_female_pct % female audience
    - demographics_male_pct   % male audience
    - demographics_other_pct  % other audience

V2 model columns (for signup rate prediction):
    - content_category, tone_of_speech_gn, fc_creator_enthusiasm_level,
      hook_category, integration_level, sponsor_placement, audience_product_fit
    - calc_pct_gender_female, calc_pct_gender_male
    - calc_pct_age_16_20, calc_pct_age_21_29, calc_pct_age_30_49, calc_pct_age_50plus
    - calc_pct_tier_1, calc_pct_tier_2, calc_pct_tier_3, calc_pct_tier_other

Optional:
    - expected_cpm             Pre-campaign CPM estimate
    - youtube_category_name    Entertainment, Gaming, Education, etc.
    - demographics_main_country  US, UK, DE, FR, etc.
    - creator_name             For display only
    - asking_price             Creator's initial price pitch
    - subscribers              Channel subscriber count
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import (
    QUANTILES, PROFITABILITY_THRESHOLD, PRE_CAMPAIGN_NUMERIC, PRE_CAMPAIGN_CATEGORICAL,
    CLASSIFIER_THRESHOLD, DEFAULT_NEW_LEVEL, DEFAULT_REBOOKING_LEVEL,
    PLATFORM_FILTER, PLAYBOOK_FEATURES,
    LOW_IAP_COUNTRIES, SUSPICIOUS_VIEW_RATIO,
)
from src.predict import load_models
from src.rebooking import build_creator_profiles, price_rebooking
from src.dynamic_benchmarks import compute_segmented_benchmarks, get_benchmark_for_creator
from src.data import load_raw, clean
from src.signup_model import (
    load_v2_models, predict_signup_rate, compute_weighted_appu,
    price_creator, flag_outlier_predictions,
)


# ══════════════════════════════════════════════════════════════
# INPUT NORMALIZATION
# ══════════════════════════════════════════════════════════════

def _parse_euro_number(val):
    """Parse numbers with European formatting: '70,000' or '314,000' or '$5,000.00'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("$", "").replace(" ", "")
    # If it looks like '70,000' (European thousands sep) or '5,000.00' (US format)
    # Remove commas that are thousands separators
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_country_pct(val):
    """Parse 'US: 50,80%' → ('US', 50.8)."""
    if pd.isna(val):
        return "Unknown", 0.0
    s = str(val).strip()
    # Format: "US: 50,80%" or "US: 50.80%"
    parts = s.split(":")
    if len(parts) != 2:
        return "Unknown", 0.0
    country = parts[0].strip()
    pct_str = parts[1].strip().replace("%", "").replace(",", ".")
    try:
        pct = float(pct_str)
    except ValueError:
        pct = 0.0
    return country, pct


def _parse_gender_pct(val):
    """Parse 'Female: 70,80%' → ('Female', 70.8)."""
    if pd.isna(val):
        return "Unknown", 0.0
    s = str(val).strip()
    parts = s.split(":")
    if len(parts) != 2:
        return "Unknown", 0.0
    gender = parts[0].strip()
    pct_str = parts[1].strip().replace("%", "").replace(",", ".")
    try:
        pct = float(pct_str)
    except ValueError:
        pct = 0.0
    return gender, pct


def _normalize_input(df):
    """
    Map hand-curated CSV columns to model-expected column names.
    Handles European number formatting and combined label:value fields.
    Only applies mappings for columns that exist in the input.
    """
    col_map = {
        "Name": "creator_name",
        "Av. views": "expected_views",
        "Subs.": "subscribers",
        "Content": "youtube_category_name",
        "Price (60s Integration)": "asking_price",
    }

    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Parse numeric columns with European/currency formatting
    for col in ["expected_views", "subscribers", "asking_price"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_euro_number)

    # Parse "-> US %" column → demographics_main_country + US pct
    us_col = "-> US %"
    if us_col in df.columns and "demographics_main_country" not in df.columns:
        parsed = df[us_col].apply(_parse_country_pct)
        df["demographics_main_country"] = parsed.apply(lambda x: x[0])
        # US % as a proxy — not directly used but informative
        df["us_pct"] = parsed.apply(lambda x: x[1])

    # Parse gender column → demographics_female_pct / male_pct
    gender_col = "-> strongest_1"
    if gender_col in df.columns:
        parsed = df[gender_col].apply(_parse_gender_pct)
        genders = parsed.apply(lambda x: x[0])
        pcts = parsed.apply(lambda x: x[1])

        if "demographics_female_pct" not in df.columns:
            df["demographics_female_pct"] = np.where(
                genders.str.lower() == "female", pcts, 100 - pcts,
            )
        if "demographics_male_pct" not in df.columns:
            df["demographics_male_pct"] = np.where(
                genders.str.lower() == "male", pcts, 100 - pcts,
            )

    return df


# ══════════════════════════════════════════════════════════════
# LAYER 1: PRE-SCREENING
# ══════════════════════════════════════════════════════════════

def _prescreen(df, original, profiles):
    """
    Apply pre-screening filters. Sets columns:
      - skip: bool (True = do not price this creator)
      - skip_reason: str (why they were skipped)
      - view_sub_ratio: float (avg views / subscribers)
      - hist_view_ratio: float (actual views / expected views from rebooking data)
    """
    original["skip"] = False
    original["skip_reason"] = ""
    original["view_sub_ratio"] = np.nan
    original["hist_view_ratio"] = np.nan

    ev = df["expected_views"].values.astype(float)
    name_col = "creator_name" if "creator_name" in df.columns else None

    for i in range(len(original)):
        reasons = []

        # --- Low-IAP country check ---
        country = df.loc[i, "demographics_main_country"] if "demographics_main_country" in df.columns else "Unknown"
        if isinstance(country, str) and country.upper() in LOW_IAP_COUNTRIES:
            reasons.append(f"primary country {country} is low-IAP")

        # --- Subscriber-to-view ratio (if subscribers available) ---
        if "subscribers" in df.columns:
            subs = pd.to_numeric(df.loc[i, "subscribers"], errors="coerce")
            if pd.notna(subs) and subs > 0 and ev[i] > 0:
                ratio = ev[i] / subs
                original.loc[i, "view_sub_ratio"] = round(ratio, 2)
                if ratio > SUSPICIOUS_VIEW_RATIO:
                    reasons.append(f"views/subs={ratio:.1f}x (threshold {SUSPICIOUS_VIEW_RATIO}x)")

        # --- View delivery check (rebooking creators only) ---
        if name_col is not None:
            creator_name = original.loc[i, name_col]
            if pd.notna(creator_name):
                key = str(creator_name).lower().strip()
                if key in profiles:
                    vr = profiles[key]["avg_view_ratio"]
                    original.loc[i, "hist_view_ratio"] = round(vr, 2)

        if reasons:
            original.loc[i, "skip"] = True
            original.loc[i, "skip_reason"] = "; ".join(reasons)

    return original


# ══════════════════════════════════════════════════════════════
# LAYER 2: CONVERSION / SIGNUP PREDICTION
# ══════════════════════════════════════════════════════════════

def _predict_conversions(df, original, clf, profiles):
    """
    Predict conversions/signups. Two paths:
      - Rebooking creators: use historical conversion rate × expected views
      - New creators (V2): use signup rate model on playbook features
      - Fallback (V1): use classifier probability as a signal

    Sets columns:
      - conversion_likelihood (V1 classifier %)
      - low_conversion_flag
      - is_rebooking, prior_campaigns, rebooking_conv_rate, rebooking_predicted_signups
      - v2_signup_rate, v2_predicted_signups, v2_high_prediction_flag
    """
    ev = df["expected_views"].values.astype(float)
    name_col = "creator_name" if "creator_name" in df.columns else None

    # --- V1 classifier (always run as a signal) ---
    feature_cols = PRE_CAMPAIGN_NUMERIC + [
        "creator_prior_campaigns", "creator_avg_iap", "creator_avg_iap_per_1k_views",
    ] + PRE_CAMPAIGN_CATEGORICAL
    X = df[[c for c in feature_cols if c in df.columns]]
    conv_prob = clf.predict_proba(X)[:, 1]
    original["conversion_likelihood"] = (conv_prob * 100).round(1)
    original["low_conversion_flag"] = conv_prob < CLASSIFIER_THRESHOLD

    # --- Rebooking: historical conversion rate ---
    original["is_rebooking"] = False
    original["prior_campaigns"] = 0
    original["rebooking_conv_rate"] = np.nan
    original["rebooking_predicted_signups"] = np.nan

    rebooking_count = 0
    for i in range(len(original)):
        if name_col is None:
            continue
        creator_name = original.loc[i, name_col]
        if pd.isna(creator_name):
            continue
        key = str(creator_name).lower().strip()
        if key not in profiles:
            continue

        p = profiles[key]
        rebooking_count += 1
        original.loc[i, "is_rebooking"] = True
        original.loc[i, "prior_campaigns"] = p["campaigns"]
        original.loc[i, "rebooking_conv_rate"] = p["conversion_rate"]
        original.loc[i, "rebooking_predicted_signups"] = round(ev[i] * p["conversion_rate"], 1)

    # --- V2 signup rate model ---
    has_v2 = all(col in df.columns for col in PLAYBOOK_FEATURES)
    original["v2_signup_rate"] = np.nan
    original["v2_predicted_signups"] = np.nan
    original["v2_high_prediction_flag"] = False

    if has_v2:
        try:
            sr_model, _, _, _ = load_v2_models()
            X_pb = df[PLAYBOOK_FEATURES].copy()
            for col in PLAYBOOK_FEATURES:
                X_pb[col] = X_pb[col].fillna("Unknown")

            pred_sr = predict_signup_rate(X_pb, sr_model)
            original["v2_signup_rate"] = pred_sr
            original["v2_predicted_signups"] = (ev * pred_sr).round(0)

            # Flag outlier predictions
            from src.data import load_playbook, build_signup_features
            df_pb = load_playbook()
            _, y_train_sr, _ = build_signup_features(load_raw(), df_pb)
            training_rates = np.expm1(y_train_sr.values)
            is_outlier, p95 = flag_outlier_predictions(pred_sr, training_rates)
            original["v2_high_prediction_flag"] = is_outlier

            n_flagged = is_outlier.sum()
            print(f"  Layer 2: V2 signup model applied ({len(original)} creators)")
            if n_flagged:
                print(f"    {n_flagged} flagged as high outlier (SR > P95 = {p95:.4%})")
        except FileNotFoundError:
            print("  Layer 2: V2 models not found — run run_pipeline.py first.")
            has_v2 = False
    else:
        print("  Layer 2: V2 playbook columns missing — V1 classifier only.")

    return original, rebooking_count, has_v2


# ══════════════════════════════════════════════════════════════
# LAYER 3: REVENUE PROJECTION (APPU × SIGNUPS → MAX PRICE)
# ══════════════════════════════════════════════════════════════

def _project_revenue(df, original, df_training, profiles, has_v2):
    """
    Convert predicted signups into revenue projections and max prices.

    Three pricing paths:
      - Rebooking: historical APPU × predicted signups (blended with benchmark)
      - V2 new: weighted APPU × V2 predicted signups
      - V1 fallback: dynamic benchmark CPMs (segmented by demographics + reach)

    Sets columns:
      - rebooking_appu, rebooking_projected_iap, rebooking_max_price, rebooking_max_cpm
      - v2_weighted_appu, v2_projected_iap, v2_max_price, v2_breakeven_cpm
      - {conservative,moderate,aggressive}_max_cpm/_max_price/_min_iap_needed
    """
    ev = df["expected_views"].values.astype(float)
    name_col = "creator_name" if "creator_name" in df.columns else None

    # --- V1 dynamic benchmarks (always computed as a reference) ---
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

        for qname in QUANTILES:
            cpm = bm[qname]
            max_price = cpm * (ev[i] / 1000)
            original.loc[i, f"{qname}_max_cpm"] = round(cpm, 2)
            original.loc[i, f"{qname}_max_price"] = round(max_price, 2)
            original.loc[i, f"{qname}_min_iap_needed"] = round(max_price * PROFITABILITY_THRESHOLD, 2)

    # --- Rebooking: historical APPU × signups, blended with benchmark ---
    original["rebooking_appu"] = np.nan
    original["rebooking_projected_iap"] = np.nan
    original["rebooking_max_price"] = np.nan
    original["rebooking_max_cpm"] = np.nan

    for i in range(len(original)):
        if not original.loc[i, "is_rebooking"]:
            continue
        if name_col is None:
            continue
        creator_name = original.loc[i, name_col]
        if pd.isna(creator_name):
            continue

        benchmark_price = original.loc[i, f"{DEFAULT_REBOOKING_LEVEL}_max_price"]
        result = price_rebooking(creator_name, ev[i], profiles, benchmark_max_price=benchmark_price)
        if result is None:
            continue

        original.loc[i, "rebooking_appu"] = result["appu"]
        original.loc[i, "rebooking_projected_iap"] = result["predicted_iap"]
        original.loc[i, "rebooking_max_price"] = result["blended_max_price"]
        original.loc[i, "rebooking_max_cpm"] = result["blended_max_cpm"]

    # --- V2: weighted APPU × V2 signups ---
    original["v2_weighted_appu"] = np.nan
    original["v2_projected_iap"] = np.nan
    original["v2_max_price"] = np.nan
    original["v2_breakeven_cpm"] = np.nan

    if has_v2:
        try:
            _, ga_appu, tier_appu, global_appu = load_v2_models()

            for i in range(len(original)):
                w_appu = compute_weighted_appu(df.iloc[i], ga_appu, tier_appu, global_appu)
                original.loc[i, "v2_weighted_appu"] = round(w_appu, 2)

                sr = original.loc[i, "v2_signup_rate"]
                if pd.notna(sr):
                    pricing = price_creator(ev[i], sr, w_appu)
                    original.loc[i, "v2_projected_iap"] = round(pricing["projected_iap"], 2)
                    original.loc[i, "v2_max_price"] = round(pricing["max_price"], 2)
                    original.loc[i, "v2_breakeven_cpm"] = round(pricing["breakeven_cpm"], 2)

            print(f"  Layer 3: APPU pricing applied ({len(original)} creators)")
        except FileNotFoundError:
            print("  Layer 3: V2 models not found — benchmark pricing only.")

    return original


# ══════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════

def _print_summary(original, has_v2):
    ev_col = "expected_views"
    name_col = "creator_name" if "creator_name" in original.columns else None
    has_asking = "asking_price" in original.columns

    print(f"\n{'═'*110}")
    print(f"  EVALUATION RESULTS")
    print(f"{'═'*110}")

    for i, row in original.iterrows():
        label = row[name_col] if name_col else f"Creator #{i+1}"
        reach = int(row[ev_col])
        is_rb = row["is_rebooking"]
        skipped = row["skip"]

        # ── Layer 1 ──
        skip_tag = f"  SKIP: {row['skip_reason']}" if skipped else ""
        vsr = row.get("view_sub_ratio")
        vsr_label = f"  Views/Subs: {vsr:.1f}x" if pd.notna(vsr) else ""

        print(f"\n  {label}  |  Reach: {reach:,}{vsr_label}{skip_tag}")

        if skipped:
            print(f"  {'─'*52}")
            continue

        # ── Layer 2: Conversion ──
        conv = row["conversion_likelihood"]
        low_conv = row.get("low_conversion_flag", False)
        flag = "  !! LOW CONVERSION SIGNAL" if low_conv else ""
        tag = f"REBOOKING ({int(row['prior_campaigns'])} prior)" if is_rb else "NEW"
        print(f"  [{tag}]  V1 Conversion: {conv}%{flag}")

        if is_rb:
            sr = row["rebooking_conv_rate"]
            signups = row["rebooking_predicted_signups"]
            vr = row["hist_view_ratio"]
            vr_label = f"  |  Hist view delivery: {vr:.0%}" if pd.notna(vr) else ""
            print(f"  Signup rate: {sr:.4%}  |  Predicted signups: {signups:,.0f}{vr_label}")

        if has_v2 and pd.notna(row.get("v2_signup_rate")):
            sr = row["v2_signup_rate"]
            signups = row["v2_predicted_signups"]
            outlier = "  !! HIGH PREDICTION" if row.get("v2_high_prediction_flag") else ""
            print(f"  V2 signup rate: {sr:.4%}  |  V2 predicted signups: {signups:,.0f}{outlier}")

        # ── Layer 3: Revenue ──
        print(f"  {'─'*52}")

        if is_rb:
            appu = row["rebooking_appu"]
            iap = row["rebooking_projected_iap"]
            mp = row["rebooking_max_price"]
            cpm = row["rebooking_max_cpm"]
            if pd.notna(appu):
                print(f"  Rebooking:  APPU ${appu:.2f}  |  Proj IAP ${iap:,.0f}  |  Max price ${mp:,.0f} (CPM ${cpm:.2f})")

        if has_v2 and pd.notna(row.get("v2_max_price")):
            appu = row["v2_weighted_appu"]
            iap = row["v2_projected_iap"]
            mp = row["v2_max_price"]
            cpm = row["v2_breakeven_cpm"]
            print(f"  V2 APPU:    APPU ${appu:.2f}  |  Proj IAP ${iap:,.2f}  |  Max price ${mp:,.2f} (CPM ${cpm:.2f})")

        if not is_rb:
            print(f"  V1 Benchmark:")
            adj = row.get("benchmark_adjustments", "")
            print(f"  Segment: {adj}")
            print(f"  {'Level':<15} {'Max CPM':>10} {'Max Price':>12} {'Min IAP':>12}")
            for lv in QUANTILES:
                print(f"  {lv:<15} ${row[f'{lv}_max_cpm']:>8,.2f} ${row[f'{lv}_max_price']:>10,.0f} ${row[f'{lv}_min_iap_needed']:>10,.0f}")

        if has_asking and pd.notna(row.get("asking_price")) and row["asking_price"] > 0:
            ask = row["asking_price"]
            best = row["best_max_price"]
            diff = row["asking_vs_best"]
            diff_pct = row["asking_vs_best_pct"]
            print(f"  Asking: ${ask:,.0f}  |  Best max: ${best:,.0f}  |  Diff: ${diff:+,.0f} ({diff_pct:+.0f}%)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def score_batch(input_path, output_path=None):
    clf, benchmarks = load_models()

    # Load training data for rebooking profiles (YouTube only)
    df_training = clean(load_raw())
    if PLATFORM_FILTER and "posting_platform" in df_training.columns:
        df_training = df_training[df_training["posting_platform"] == PLATFORM_FILTER].copy()
    profiles = build_creator_profiles(df_training)

    df = pd.read_csv(input_path)

    # ── Normalize input columns (handles hand-curated CSVs) ──
    df = _normalize_input(df)

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

    # Creator history columns (zero for new creators)
    df["creator_prior_campaigns"] = 0
    df["creator_avg_iap"] = 0.0
    df["creator_avg_iap_per_1k_views"] = 0.0

    # Copy key columns from working df to original for output
    original["expected_views"] = df["expected_views"]

    # ── LAYER 1: Pre-screening ──
    print("Layer 1: Pre-screening...")
    original = _prescreen(df, original, profiles)
    n_skip = original["skip"].sum()
    print(f"  {n_skip} skipped, {len(original) - n_skip} proceed to scoring")

    # ── LAYER 2: Conversion / signup prediction ──
    print("\nLayer 2: Conversion prediction...")
    original, rebooking_count, has_v2 = _predict_conversions(df, original, clf, profiles)
    print(f"  Rebookings: {rebooking_count}  |  New: {len(original) - rebooking_count}")

    # ── LAYER 3: Revenue projection ──
    print("\nLayer 3: Revenue projection...")
    original = _project_revenue(df, original, df_training, profiles, has_v2)

    # Asking price comparison (vs rebooking for returning, moderate benchmark for new)
    has_asking = "asking_price" in original.columns
    if has_asking:
        asking = pd.to_numeric(original["asking_price"], errors="coerce")
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
    print(f"\nScored {len(original)} creators -> {output_path}")

    _print_summary(original, has_v2)

    return original


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    score_batch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
