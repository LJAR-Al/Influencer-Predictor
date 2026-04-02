"""Data loading and cleaning for influencer pricing pipeline."""
import json
import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    SUPABASE_URL, SUPABASE_ANON_KEY, TABLE_NAME,
    IAP_COL, MIN_COST, MIN_EXPECTED_VIEWS,
    PRE_CAMPAIGN_NUMERIC, PRE_CAMPAIGN_CATEGORICAL,
    PLATFORM_FILTER, PLAYBOOK_TABLE, PLAYBOOK_FEATURES,
    DEMO_AGE_COLS, DEMO_GENDER_COLS,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_from_supabase(save=True):
    """Pull all rows from Supabase, optionally save raw JSON."""
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Prefer": "count=exact",
    }
    all_rows = []
    offset = 0
    batch = 1000
    while True:
        r = requests.get(
            f"{url}?limit={batch}&offset={offset}&order=published_at.desc",
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        all_rows.extend(data)
        if len(data) < batch:
            break
        offset += batch

    if save:
        DATA_DIR.mkdir(exist_ok=True)
        with open(DATA_DIR / "raw_campaigns.json", "w") as f:
            json.dump(all_rows, f)

    return pd.DataFrame(all_rows)


def load_raw(path=None):
    """Load raw JSON from disk."""
    path = path or DATA_DIR / "raw_campaigns.json"
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def _extract_creator_id(campaign_name):
    """Extract creator identifier from campaign name like 'BLAIRE-WHITE_YT_R3'."""
    if not isinstance(campaign_name, str):
        return None
    match = re.match(r'^(.+?)(?:_(?:YT|IGR|TT|IG|IGSTORY|IGREEL)[_].*)?$', campaign_name)
    if match:
        return match.group(1)
    return campaign_name


def clean(df):
    """Clean and type-cast the raw dataframe."""
    df = df.copy()

    # Normalize platform names
    df["posting_platform"] = df["posting_platform"].str.strip().str.title()

    # Extract creator ID
    df["creator_id"] = df["clean_campaign_name"].apply(_extract_creator_id)

    # Parse publish date
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    # Cast numeric columns
    numeric_cast = [
        IAP_COL, "campaign_cost_cleaned",
        "expected_views", "expected_cpm",
        "view_count", "youtube_like_count", "youtube_comment_count",
        "youtube_duration_ss",
        "demographics_female %", "demographics_male %", "demographics_other %",
        "total_signups", "total_purchases",
        "calc_pct_tier_1", "calc_pct_tier_2", "calc_pct_tier_3",
    ]
    for col in numeric_cast:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename awkward column names
    df = df.rename(columns={
        "demographics_female %": "demographics_female_pct",
        "demographics_male %": "demographics_male_pct",
        "demographics_other %": "demographics_other_pct",
    })

    return df


def _add_creator_history(df):
    """
    Add historical creator performance features.
    Uses published_at ordering to prevent data leakage.
    """
    df = df.sort_values("published_at").reset_index(drop=True)

    df["creator_prior_campaigns"] = 0
    df["creator_avg_iap"] = np.nan
    df["creator_avg_iap_per_1k_views"] = np.nan

    for idx in df.index:
        cid = df.loc[idx, "creator_id"]
        pub_date = df.loc[idx, "published_at"]
        if pd.isna(cid) or pd.isna(pub_date):
            continue

        prior = df[(df["creator_id"] == cid) & (df["published_at"] < pub_date)]
        if len(prior) == 0:
            continue

        prior_rev = prior[IAP_COL].dropna()
        if len(prior_rev) == 0:
            continue

        df.loc[idx, "creator_prior_campaigns"] = len(prior_rev)
        df.loc[idx, "creator_avg_iap"] = prior_rev.mean()

        prior_ev = prior["expected_views"].replace(0, np.nan)
        rev_per_1k = (prior[IAP_COL] / prior_ev * 1000).dropna()
        if len(rev_per_1k) > 0:
            df.loc[idx, "creator_avg_iap_per_1k_views"] = rev_per_1k.mean()

    return df


def build_features(df):
    """
    Build feature matrix and targets for the pricing model.

    Features: pre-campaign data only (demographics, reach, CPM, category, platform)
    Targets: binary conversion + log IAP (d7 users at d14)
    """
    df = df.copy()

    # Filter to YouTube only
    if PLATFORM_FILTER and "posting_platform" in df.columns:
        df = df[df["posting_platform"] == PLATFORM_FILTER].copy()

    # Filter: need valid cost, IAP, and expected views
    cost = pd.to_numeric(df.get("campaign_cost_cleaned", 0), errors="coerce")
    iap = df[IAP_COL]
    ev = df["expected_views"]
    mask = (cost >= MIN_COST) & iap.notna() & (ev >= MIN_EXPECTED_VIEWS)
    df = df[mask].copy()

    # Cap IAP outliers at 99th percentile
    rev_cap = df[IAP_COL].quantile(0.99)
    df[IAP_COL] = df[IAP_COL].clip(upper=rev_cap)

    # Binary conversion label (for two-stage model)
    df["converted"] = (df[IAP_COL] > 0).astype(int)

    # Log-transformed IAP target
    df["log_iap"] = np.log1p(df[IAP_COL])

    # Add creator history (leak-free)
    df = _add_creator_history(df)

    # Assemble features
    history_numeric = [
        "creator_prior_campaigns",
        "creator_avg_iap",
        "creator_avg_iap_per_1k_views",
    ]
    all_numeric = PRE_CAMPAIGN_NUMERIC + history_numeric
    all_features = all_numeric + PRE_CAMPAIGN_CATEGORICAL
    existing = [c for c in all_features if c in df.columns]

    X = df[existing].copy()
    y_log = df["log_iap"].copy()
    y_binary = df["converted"].copy()

    # Fill missing numerics with median
    for col in all_numeric:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # Fill missing categoricals with "Unknown"
    for col in PRE_CAMPAIGN_CATEGORICAL:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown")

    return X, y_log, y_binary, df


# ── V2: Playbook data loading and signup-rate feature building ──

def fetch_playbook_from_supabase(save=True):
    """Pull all rows from video_playbook_results table."""
    url = f"{SUPABASE_URL}/rest/v1/{PLAYBOOK_TABLE}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    all_rows = []
    offset = 0
    batch = 1000
    while True:
        r = requests.get(
            f"{url}?limit={batch}&offset={offset}&data_status=eq.analyzed",
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        all_rows.extend(data)
        if len(data) < batch:
            break
        offset += batch

    if save:
        DATA_DIR.mkdir(exist_ok=True)
        with open(DATA_DIR / "raw_playbook.json", "w") as f:
            json.dump(all_rows, f)

    return pd.DataFrame(all_rows)


def load_playbook(path=None):
    """Load playbook JSON from disk."""
    path = path or DATA_DIR / "raw_playbook.json"
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def build_signup_features(df_campaigns, df_playbook):
    """
    Join campaigns with playbook data and build features for signup-rate model.

    Returns: X (playbook features), y (log signup rate), df_joined (full joined data)
    """
    df_c = df_campaigns.copy()
    df_p = df_playbook.copy()

    # Filter to YouTube
    if PLATFORM_FILTER and "posting_platform" in df_c.columns:
        df_c = df_c[df_c["posting_platform"] == PLATFORM_FILTER].copy()

    # Cast numeric columns needed for joining/filtering
    for col in ["campaign_cost_cleaned", "expected_views", "view_count",
                 "total_signups", IAP_COL] + list(DEMO_AGE_COLS.keys()) + list(DEMO_GENDER_COLS.keys()):
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

    # Join on youtube_video_id
    df_joined = df_c.merge(
        df_p[["youtube_video_id"] + PLAYBOOK_FEATURES],
        on="youtube_video_id",
        how="inner",
    )

    # Filter: need cost, views, and demographic data
    mask = (
        (df_joined["campaign_cost_cleaned"] > 0)
        & (df_joined["expected_views"] > 0)
        & (df_joined["view_count"] > 0)
        & (df_joined["total_signups"] >= 0)
        & (
            (df_joined.get("calc_pct_age_16_20", 0) > 0)
            | (df_joined.get("calc_pct_age_21_29", 0) > 0)
        )
    )
    df_joined = df_joined[mask].copy()

    # Compute target
    df_joined["signup_rate"] = df_joined["total_signups"] / df_joined["view_count"]
    df_joined["log_signup_rate"] = np.log1p(df_joined["signup_rate"])

    # Build feature matrix
    X = df_joined[PLAYBOOK_FEATURES].copy()
    for col in PLAYBOOK_FEATURES:
        X[col] = X[col].fillna("Unknown")

    y = df_joined["log_signup_rate"]

    return X, y, df_joined
