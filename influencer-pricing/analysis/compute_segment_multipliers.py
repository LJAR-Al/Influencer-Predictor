"""
Compute conversion propensity and APPU multipliers per demographic segment.

Uses actual Freecash converter data to determine:
1. Which demographics convert at higher/lower rates
2. Which demographics generate higher/lower APPU
3. How audience composition should be adjusted when predicting revenue
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Load campaign data
with open(DATA_DIR / "raw_campaigns.json") as f:
    df = pd.DataFrame(json.load(f))

# Cast columns
int_cols = [
    "total_signups", "gender_female", "gender_male",
    "gender_not_specified",
    "age_group_from_16_to_20", "age_group_from_21_to_29",
    "age_group_from_30_to_49", "age_group_from_50",
    "age_group_not_specified", "age_group_null",
    "tier_1_signups", "tier_2_signups", "tier_3_signups", "tier_other_signups",
]
float_cols = ["d7_user_purchases_by_d14_publish_date", "view_count"]

for c in int_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
for c in float_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Filter: need signups and views
mask = (df["total_signups"] > 20) & (df["view_count"] > 0)
df = df[mask].copy()
print(f"Campaigns with >20 signups: {len(df)}")
print(f"Total signups: {df['total_signups'].sum():,}")
print(f"Total IAP (d7@d14): ${df['d7_user_purchases_by_d14_publish_date'].sum():,.2f}")

# ══════════════════════════════════════════════════════════════
# GENDER ANALYSIS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GENDER: APPU by converter gender mix")
print(f"{'='*70}")

total_f = df["gender_female"].sum()
total_m = df["gender_male"].sum()
total_all = df["total_signups"].sum()
total_iap = df["d7_user_purchases_by_d14_publish_date"].sum()

print(f"\nOverall: {total_f:,} female + {total_m:,} male = {total_all:,} signups")
print(f"Overall APPU: ${total_iap / total_all:.2f}")

# Per-campaign: compute APPU and correlate with gender mix
df["appu"] = df["d7_user_purchases_by_d14_publish_date"] / df["total_signups"]
df["female_pct"] = df["gender_female"] / df["total_signups"] * 100
df["male_pct"] = df["gender_male"] / df["total_signups"] * 100

# Bucket by converter female %
bins = [0, 20, 40, 60, 80, 101]
labels = ["0-20% F", "20-40% F", "40-60% F", "60-80% F", "80-100% F"]
df["gender_bucket"] = pd.cut(df["female_pct"], bins=bins, labels=labels, right=False)

print(f"\n{'Bucket':<15} {'Campaigns':>10} {'Signups':>10} {'IAP':>12} {'APPU':>8} {'Signup/View':>12}")
print(f"{'-'*70}")
for bucket in labels:
    sub = df[df["gender_bucket"] == bucket]
    if len(sub) == 0:
        continue
    s = sub["total_signups"].sum()
    iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
    v = sub["view_count"].sum()
    print(f"{bucket:<15} {len(sub):>10} {s:>10,} ${iap:>10,.0f} ${iap/s:>6.2f} {s/v*100:>10.3f}%")


# ══════════════════════════════════════════════════════════════
# AGE ANALYSIS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("AGE: APPU by converter age distribution")
print(f"{'='*70}")

df["pct_16_20"] = df["age_group_from_16_to_20"] / df["total_signups"] * 100
df["pct_21_29"] = df["age_group_from_21_to_29"] / df["total_signups"] * 100
df["pct_30_49"] = df["age_group_from_30_to_49"] / df["total_signups"] * 100
df["pct_50plus"] = df["age_group_from_50"] / df["total_signups"] * 100

# Find dominant age group per campaign
df["dominant_age"] = df[["pct_16_20", "pct_21_29", "pct_30_49", "pct_50plus"]].idxmax(axis=1)
age_map = {
    "pct_16_20": "16-20 dominant",
    "pct_21_29": "21-29 dominant",
    "pct_30_49": "30-49 dominant",
    "pct_50plus": "50+ dominant",
}
df["dominant_age"] = df["dominant_age"].map(age_map)

print(f"\n{'Age group':<20} {'Campaigns':>10} {'Signups':>10} {'IAP':>12} {'APPU':>8} {'Signup/View':>12}")
print(f"{'-'*75}")
for age in ["16-20 dominant", "21-29 dominant", "30-49 dominant", "50+ dominant"]:
    sub = df[df["dominant_age"] == age]
    if len(sub) == 0:
        continue
    s = sub["total_signups"].sum()
    iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
    v = sub["view_count"].sum()
    print(f"{age:<20} {len(sub):>10} {s:>10,} ${iap:>10,.0f} ${iap/s:>6.2f} {s/v*100:>10.3f}%")


# ══════════════════════════════════════════════════════════════
# GEO TIER ANALYSIS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GEO: APPU by converter tier distribution")
print(f"{'='*70}")

df["pct_tier1"] = df["tier_1_signups"] / df["total_signups"] * 100
df["pct_tier3"] = df["tier_3_signups"] / df["total_signups"] * 100

# Bucket by tier 1 %
bins_t = [0, 20, 50, 80, 101]
labels_t = ["<20% T1", "20-50% T1", "50-80% T1", "80-100% T1"]
df["tier_bucket"] = pd.cut(df["pct_tier1"], bins=bins_t, labels=labels_t, right=False)

print(f"\n{'Bucket':<15} {'Campaigns':>10} {'Signups':>10} {'IAP':>12} {'APPU':>8} {'Signup/View':>12}")
print(f"{'-'*70}")
for bucket in labels_t:
    sub = df[df["tier_bucket"] == bucket]
    if len(sub) == 0:
        continue
    s = sub["total_signups"].sum()
    iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
    v = sub["view_count"].sum()
    print(f"{bucket:<15} {len(sub):>10} {s:>10,} ${iap:>10,.0f} ${iap/s:>6.2f} {s/v*100:>10.3f}%")


# ══════════════════════════════════════════════════════════════
# CROSS-TABULATION: Gender × Age
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CROSS-TAB: Gender skew × Age skew → APPU")
print(f"{'='*70}")

df["gender_skew"] = pd.cut(
    df["female_pct"],
    bins=[0, 40, 60, 101],
    labels=["Male-heavy", "Balanced", "Female-heavy"],
    right=False,
)

print(f"\n{'Gender':<15} {'Age':<20} {'N':>5} {'Signups':>10} {'APPU':>8} {'Signup/View':>12}")
print(f"{'-'*75}")
for gs in ["Male-heavy", "Balanced", "Female-heavy"]:
    for age in ["16-20 dominant", "21-29 dominant", "30-49 dominant", "50+ dominant"]:
        sub = df[(df["gender_skew"] == gs) & (df["dominant_age"] == age)]
        if len(sub) < 2:
            continue
        s = sub["total_signups"].sum()
        iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
        v = sub["view_count"].sum()
        print(f"{gs:<15} {age:<20} {len(sub):>5} {s:>10,} ${iap/s:>6.2f} {s/v*100:>10.3f}%")


# ══════════════════════════════════════════════════════════════
# COMPUTE MULTIPLIERS (vs global average)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("MULTIPLIERS (relative to global average)")
print(f"{'='*70}")

global_appu = total_iap / total_all
global_sr = total_all / df["view_count"].sum()

print(f"\nGlobal APPU: ${global_appu:.2f}")
print(f"Global signup rate: {global_sr:.4%}")

print(f"\n--- APPU multipliers ---")
for gs in ["Male-heavy", "Balanced", "Female-heavy"]:
    sub = df[df["gender_skew"] == gs]
    s = sub["total_signups"].sum()
    iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
    appu = iap / s if s > 0 else 0
    print(f"  {gs:<15} APPU ${appu:.2f}  mult: {appu/global_appu:.2f}x  (n={len(sub)})")

for age in ["16-20 dominant", "21-29 dominant", "30-49 dominant", "50+ dominant"]:
    sub = df[df["dominant_age"] == age]
    s = sub["total_signups"].sum()
    iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
    appu = iap / s if s > 0 else 0
    print(f"  {age:<20} APPU ${appu:.2f}  mult: {appu/global_appu:.2f}x  (n={len(sub)})")

for bucket in labels_t:
    sub = df[df["tier_bucket"] == bucket]
    s = sub["total_signups"].sum()
    iap = sub["d7_user_purchases_by_d14_publish_date"].sum()
    appu = iap / s if s > 0 else 0
    print(f"  {bucket:<15} APPU ${appu:.2f}  mult: {appu/global_appu:.2f}x  (n={len(sub)})")

print(f"\n--- Signup rate multipliers ---")
for gs in ["Male-heavy", "Balanced", "Female-heavy"]:
    sub = df[df["gender_skew"] == gs]
    s = sub["total_signups"].sum()
    v = sub["view_count"].sum()
    sr = s / v if v > 0 else 0
    print(f"  {gs:<15} SR {sr:.4%}  mult: {sr/global_sr:.2f}x  (n={len(sub)})")

for age in ["16-20 dominant", "21-29 dominant", "30-49 dominant", "50+ dominant"]:
    sub = df[df["dominant_age"] == age]
    s = sub["total_signups"].sum()
    v = sub["view_count"].sum()
    sr = s / v if v > 0 else 0
    print(f"  {age:<20} SR {sr:.4%}  mult: {sr/global_sr:.2f}x  (n={len(sub)})")
