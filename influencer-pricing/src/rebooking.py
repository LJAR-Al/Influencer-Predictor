"""Rebooking pricing: use actual creator performance data for returning creators."""
import numpy as np
import pandas as pd

from src.config import IAP_COL, PROFITABILITY_THRESHOLD


# Confidence weight for rebooking data based on number of prior campaigns.
# With high variance across campaigns (median CV = 1.41), a single campaign
# is unreliable. We blend rebooking price with benchmark price:
#   blended = weight * rebooking_price + (1 - weight) * benchmark_price
# Weight increases with more campaigns but saturates.
def _rebooking_weight(n_campaigns):
    """Weight for rebooking data: 0.5 at 1 campaign, 0.8 at 3, ~1.0 at 5+."""
    return min(1.0, 0.5 + 0.15 * (n_campaigns - 1))


def build_creator_profiles(df_training):
    """
    Build performance profiles for all creators in training data.

    Conversion rate is stored per actual view. Data analysis confirms
    signups scale linearly with views (power ≈ 1.0), so projecting
    conv_rate_per_view × new_expected_views is the correct normalization.
    """
    cost = pd.to_numeric(df_training.get("campaign_cost_cleaned", 0), errors="coerce")
    rev = df_training[IAP_COL]
    ev = pd.to_numeric(df_training.get("expected_views", 0), errors="coerce")
    vc = pd.to_numeric(df_training.get("view_count", 0), errors="coerce")
    signups = pd.to_numeric(df_training.get("total_signups", 0), errors="coerce")

    mask = (cost > 0) & (ev > 0) & rev.notna() & (vc > 0)
    d = df_training[mask].copy()

    profiles = {}
    for name in d["youtube_channel_name"].dropna().unique():
        rows = d[d["youtube_channel_name"] == name]
        total_views = vc[rows.index].sum()
        total_signups = signups[rows.index].sum()
        total_iap = rev[rows.index].sum()
        total_cost = cost[rows.index].sum()
        total_ev = ev[rows.index].sum()

        if total_views == 0:
            continue

        conv_rate = total_signups / total_views if total_views > 0 else 0
        appu = total_iap / total_signups if total_signups > 0 else 0
        avg_view_ratio = total_views / total_ev if total_ev > 0 else 1.0

        profiles[name.lower()] = {
            "conversion_rate": float(conv_rate),
            "appu": float(appu),
            "avg_view_ratio": float(avg_view_ratio),
            "total_iap": float(total_iap),
            "total_views": int(total_views),
            "total_expected_views": int(total_ev),
            "total_signups": int(total_signups),
            "campaigns": len(rows),
        }

    return profiles


def price_rebooking(creator_name, expected_views, profiles, benchmark_max_price=None):
    """
    Price a returning creator based on their actual performance,
    blended with the benchmark price based on confidence.

    With 1 prior campaign (high variance), we weight rebooking data 50%
    and benchmark 50%. With 5+ campaigns, rebooking dominates.

    Args:
        benchmark_max_price: The benchmark-based max price for this creator.
            If provided, the output blends rebooking and benchmark.
    """
    key = creator_name.lower().strip()
    if key not in profiles:
        return None

    p = profiles[key]
    ev = float(expected_views)

    conv_rate = p["conversion_rate"]
    appu = p["appu"]

    # Projected performance at new expected views
    predicted_signups = ev * conv_rate
    predicted_iap = predicted_signups * appu
    rebooking_max_price = predicted_iap / PROFITABILITY_THRESHOLD
    rebooking_max_cpm = rebooking_max_price / (ev / 1000) if ev > 0 else 0

    # Confidence-weighted blend
    weight = _rebooking_weight(p["campaigns"])

    if benchmark_max_price is not None and benchmark_max_price > 0:
        blended_max_price = weight * rebooking_max_price + (1 - weight) * benchmark_max_price
        blended_max_cpm = blended_max_price / (ev / 1000) if ev > 0 else 0
    else:
        blended_max_price = rebooking_max_price
        blended_max_cpm = rebooking_max_cpm

    return {
        "is_rebooking": True,
        "prior_campaigns": p["campaigns"],
        "rebooking_weight": weight,
        "conversion_rate": conv_rate,
        "appu": appu,
        "avg_view_ratio": p["avg_view_ratio"],
        "historical_views": p["total_views"],
        "historical_expected_views": p["total_expected_views"],
        "predicted_signups": round(predicted_signups, 1),
        "predicted_iap": round(predicted_iap, 2),
        "rebooking_max_price": round(rebooking_max_price, 2),
        "rebooking_max_cpm": round(rebooking_max_cpm, 2),
        "blended_max_price": round(blended_max_price, 2),
        "blended_max_cpm": round(blended_max_cpm, 2),
    }
