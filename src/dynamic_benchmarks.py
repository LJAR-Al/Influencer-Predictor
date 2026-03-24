"""Dynamic CPM benchmarks segmented by country group and audience gender skew."""
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


def compute_segmented_benchmarks(df_training):
    """
    Compute CPM benchmarks segmented by country group × gender skew.

    Falls back to broader segments when a specific cross has too few
    profitable campaigns (< MIN_SEGMENT_SIZE).

    Returns dict of {(country_group, gender_skew): {quantile_name: cpm_value}}.
    Also includes "global" key as ultimate fallback.
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

    def _quantiles(series):
        return {name: float(series.quantile(q)) for name, q in QUANTILES.items()}

    benchmarks = {}

    # Global
    benchmarks["global"] = _quantiles(prof["paid_cpm"])
    benchmarks["global"]["count"] = len(prof)

    # By country group
    for cg in ["US", "Europe", "Other"]:
        sub = prof[prof["country_group"] == cg]
        if len(sub) >= MIN_SEGMENT_SIZE:
            benchmarks[f"country:{cg}"] = _quantiles(sub["paid_cpm"])
            benchmarks[f"country:{cg}"]["count"] = len(sub)

    # By gender skew
    for gs in ["Male-skewed", "Balanced", "Female-skewed"]:
        sub = prof[prof["gender_skew"] == gs]
        if len(sub) >= MIN_SEGMENT_SIZE:
            benchmarks[f"gender:{gs}"] = _quantiles(sub["paid_cpm"])
            benchmarks[f"gender:{gs}"]["count"] = len(sub)

    # Cross: country × gender
    for cg in ["US", "Europe", "Other"]:
        for gs in ["Male-skewed", "Balanced", "Female-skewed"]:
            sub = prof[(prof["country_group"] == cg) & (prof["gender_skew"] == gs)]
            if len(sub) >= MIN_SEGMENT_SIZE:
                benchmarks[f"{cg}:{gs}"] = _quantiles(sub["paid_cpm"])
                benchmarks[f"{cg}:{gs}"]["count"] = len(sub)

    return benchmarks


def get_benchmark_for_creator(country, female_pct, benchmarks):
    """
    Look up the best available benchmark for a creator's profile.

    Tries most specific (country × gender) first, then falls back to
    country-only, gender-only, and finally global.

    Returns (benchmark_dict, segment_name).
    """
    cg = _country_group(country)
    gs = _gender_skew(female_pct)

    # Try most specific first
    key = f"{cg}:{gs}"
    if key in benchmarks:
        return benchmarks[key], key

    # Fall back to country
    key = f"country:{cg}"
    if key in benchmarks:
        return benchmarks[key], key

    # Fall back to gender
    key = f"gender:{gs}"
    if key in benchmarks:
        return benchmarks[key], key

    # Global fallback
    return benchmarks["global"], "global"
