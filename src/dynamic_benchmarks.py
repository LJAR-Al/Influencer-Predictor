"""Dynamic CPM benchmarks with reach tier, category, country, and gender adjustments."""
import numpy as np
import pandas as pd

from src.config import REVENUE_COL, PROFITABILITY_THRESHOLD, QUANTILES

# Minimum segment size — fall back to parent segment if too few samples
MIN_SEGMENT_SIZE = 5


def _country_group(country):
    if country == "US":
        return "US"
    if country in ("PL", "FR", "UK", "DE", "NL", "IT", "ES", "SE", "NO", "DK", "AT", "CH", "BE"):
        return "Europe"
    return "Other"


def _gender_skew(female_pct):
    if pd.isna(female_pct):
        return "Unknown"
    if female_pct < 40:
        return "Male-skewed"
    if female_pct <= 60:
        return "Balanced"
    return "Female-skewed"


def _reach_tier(views):
    if pd.isna(views) or views <= 0:
        return "Unknown"
    if views < 50000:
        return "<50k"
    if views < 100000:
        return "50-100k"
    if views < 200000:
        return "100-200k"
    if views < 500000:
        return "200-500k"
    return "500k+"


def compute_segmented_benchmarks(df_training):
    """
    Compute CPM benchmarks segmented by:
      - country group × gender skew (primary segmentation)
      - reach tier multipliers (applied on top)
      - category multipliers (applied on top)

    Returns dict with segment benchmarks, reach multipliers, and category multipliers.
    """
    cost = pd.to_numeric(df_training.get("campaign_cost_cleaned", 0), errors="coerce")
    rev = df_training[REVENUE_COL]
    ev = pd.to_numeric(df_training.get("expected_views", 0), errors="coerce")
    roi = rev / cost

    mask = (roi >= PROFITABILITY_THRESHOLD) & (cost > 0) & (ev > 0)
    prof = df_training[mask].copy()
    prof["paid_cpm"] = cost[prof.index] / (ev[prof.index] / 1000)
    prof["country_group"] = prof["demographics_main_country"].apply(_country_group)
    fem = pd.to_numeric(prof.get("demographics_female_pct", 0), errors="coerce")
    prof["gender_skew"] = fem.apply(_gender_skew)
    prof["reach_tier"] = ev[prof.index].apply(_reach_tier)

    def _quantiles(series):
        return {name: float(series.quantile(q)) for name, q in QUANTILES.items()}

    global_median = float(prof["paid_cpm"].median())

    # --- Country × gender segment benchmarks ---
    segments = {}

    segments["global"] = _quantiles(prof["paid_cpm"])
    segments["global"]["count"] = len(prof)

    for cg in ["US", "Europe", "Other"]:
        sub = prof[prof["country_group"] == cg]
        if len(sub) >= MIN_SEGMENT_SIZE:
            segments[f"country:{cg}"] = _quantiles(sub["paid_cpm"])
            segments[f"country:{cg}"]["count"] = len(sub)

    for gs in ["Male-skewed", "Balanced", "Female-skewed"]:
        sub = prof[prof["gender_skew"] == gs]
        if len(sub) >= MIN_SEGMENT_SIZE:
            segments[f"gender:{gs}"] = _quantiles(sub["paid_cpm"])
            segments[f"gender:{gs}"]["count"] = len(sub)

    for cg in ["US", "Europe", "Other"]:
        for gs in ["Male-skewed", "Balanced", "Female-skewed"]:
            sub = prof[(prof["country_group"] == cg) & (prof["gender_skew"] == gs)]
            if len(sub) >= MIN_SEGMENT_SIZE:
                segments[f"{cg}:{gs}"] = _quantiles(sub["paid_cpm"])
                segments[f"{cg}:{gs}"]["count"] = len(sub)

    # --- Reach tier multipliers ---
    # Ratio of tier median CPM vs global median CPM
    reach_multipliers = {}
    for tier in ["<50k", "50-100k", "100-200k", "200-500k", "500k+"]:
        sub = prof[prof["reach_tier"] == tier]
        if len(sub) >= 3:
            tier_median = float(sub["paid_cpm"].median())
            reach_multipliers[tier] = tier_median / global_median
        else:
            reach_multipliers[tier] = 1.0

    # --- Category multipliers ---
    category_multipliers = {}
    for cat in prof["youtube_category_name"].dropna().unique():
        sub = prof[prof["youtube_category_name"] == cat]
        if len(sub) >= 3:
            cat_median = float(sub["paid_cpm"].median())
            category_multipliers[cat] = cat_median / global_median

    return {
        "segments": segments,
        "reach_multipliers": reach_multipliers,
        "category_multipliers": category_multipliers,
        "global_median": global_median,
        "profitable_count": len(prof),
    }


def get_benchmark_for_creator(country, female_pct, expected_views, category, benchmarks):
    """
    Look up the best available benchmark for a creator's profile,
    then apply reach tier and category multipliers.

    Returns (adjusted_benchmark_dict, segment_name, adjustments_str).
    """
    segments = benchmarks["segments"]
    reach_mults = benchmarks["reach_multipliers"]
    cat_mults = benchmarks["category_multipliers"]

    cg = _country_group(country)
    gs = _gender_skew(female_pct)
    rt = _reach_tier(expected_views)

    # Find base segment (most specific first)
    for key in [f"{cg}:{gs}", f"country:{cg}", f"gender:{gs}", "global"]:
        if key in segments:
            base = segments[key]
            seg_name = key
            break

    # Apply reach multiplier (dampened: sqrt to avoid overadjusting
    # since segment CPMs already partially capture these patterns)
    raw_reach = reach_mults.get(rt, 1.0)
    reach_mult = 1.0 + (raw_reach - 1.0) * 0.5  # Half-strength

    # Apply category multiplier (dampened similarly)
    raw_cat = cat_mults.get(category, 1.0) if pd.notna(category) else 1.0
    cat_mult = 1.0 + (raw_cat - 1.0) * 0.5  # Half-strength

    combined_mult = reach_mult * cat_mult

    # Adjust base CPMs
    adjusted = {}
    for qname in QUANTILES:
        adjusted[qname] = base[qname] * combined_mult
    adjusted["count"] = base.get("count", 0)

    # Build adjustments description
    parts = [seg_name]
    if abs(reach_mult - 1.0) > 0.01:
        parts.append(f"reach:{rt}={reach_mult:.2f}x")
    if abs(cat_mult - 1.0) > 0.01:
        parts.append(f"cat={cat_mult:.2f}x")
    adjustments = " | ".join(parts)

    return adjusted, seg_name, adjustments
