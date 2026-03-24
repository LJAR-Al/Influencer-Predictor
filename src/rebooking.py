"""Rebooking pricing: use actual creator performance data for returning creators."""
import numpy as np
import pandas as pd

from src.config import REVENUE_COL, PROFITABILITY_THRESHOLD


def build_creator_profiles(df_training):
    """
    Build performance profiles for all creators in training data.

    Conversion rate is stored per actual view. Data analysis confirms
    signups scale linearly with views (power ≈ 1.0), so projecting
    conv_rate_per_view × new_expected_views is the correct normalization.

    Returns dict of {creator_name_lower: profile_dict}.
    """
    cost = pd.to_numeric(df_training.get("campaign_cost_cleaned", 0), errors="coerce")
    rev = df_training[REVENUE_COL]
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

        # Conversion rate per actual view (linear scaling confirmed by data)
        conv_rate = total_signups / total_views if total_views > 0 else 0
        appu = total_iap / total_signups if total_signups > 0 else 0
        avg_view_ratio = total_views / total_ev if total_ev > 0 else 1.0
        realized_cpm = total_cost / (total_views / 1000)
        expected_cpm = total_cost / (total_ev / 1000) if total_ev > 0 else 0

        profiles[name.lower()] = {
            "conversion_rate": float(conv_rate),
            "appu": float(appu),
            "realized_cpm": float(realized_cpm),
            "expected_cpm": float(expected_cpm),
            "avg_view_ratio": float(avg_view_ratio),
            "total_iap": float(total_iap),
            "total_views": int(total_views),
            "total_expected_views": int(total_ev),
            "total_signups": int(total_signups),
            "campaigns": len(rows),
        }

    return profiles


def price_rebooking(creator_name, expected_views, profiles):
    """
    Price a returning creator based on their actual performance.

    Uses conversion_rate_per_actual_view × new_expected_views.
    Data confirms signups scale linearly with views (power ≈ 1.0),
    so this is the correct projection regardless of whether the
    creator previously under- or over-delivered on views.

    Returns dict with pricing or None if creator not found.
    """
    key = creator_name.lower().strip()
    if key not in profiles:
        return None

    p = profiles[key]
    ev = float(expected_views)

    conv_rate = p["conversion_rate"]
    appu = p["appu"]

    # Projected performance at new expected views
    # Linear scaling: signups = conv_rate_per_view × views
    predicted_signups = ev * conv_rate
    predicted_iap = predicted_signups * appu
    max_price = predicted_iap / PROFITABILITY_THRESHOLD
    max_cpm = max_price / (ev / 1000) if ev > 0 else 0

    return {
        "is_rebooking": True,
        "prior_campaigns": p["campaigns"],
        "conversion_rate": conv_rate,
        "appu": appu,
        "avg_view_ratio": p["avg_view_ratio"],
        "historical_views": p["total_views"],
        "historical_expected_views": p["total_expected_views"],
        "predicted_signups": round(predicted_signups, 1),
        "predicted_iap": round(predicted_iap, 2),
        "max_price": round(max_price, 2),
        "max_cpm": round(max_cpm, 2),
    }
