"""Run the influencer pricing pipeline (V1 classifier + V2 signup rate model)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import load_raw, clean, build_features, load_playbook, build_signup_features
from src.model import train_classifier, compute_profitable_benchmarks, get_feature_importance, save_results
from src.predict import save_models, format_scorecard
from src.signup_model import (
    build_appu_lookup, compute_weighted_appu, train_signup_model,
    predict_signup_rate, get_signup_feature_importance,
    price_creator, format_scorecard_v2, save_v2_models,
)
from src.config import QUANTILES, PROFITABILITY_THRESHOLD, IAP_COL

DATA_DIR = Path(__file__).parent / "data"

# ══════════════════════════════════════════════════════════════
# V1: CLASSIFIER + BENCHMARK CPMs (preserved)
# ══════════════════════════════════════════════════════════════

print("Loading data...")
df_raw = load_raw()
df = clean(df_raw)
print(f"  Raw rows: {len(df_raw)}")
if "posting_platform" in df.columns:
    print(f"  Platform split: {df['posting_platform'].value_counts().to_dict()}")

print("Building features...")
X, y_log, y_binary, df_model = build_features(df)
print(f"  Modeling rows:  {len(X)}")
print(f"  Converters:     {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
print(f"  Non-converters: {(y_binary == 0).sum()} ({(1 - y_binary.mean())*100:.1f}%)")

print("\nComputing profitable campaign benchmarks...")
benchmarks = compute_profitable_benchmarks(df_model)
print(f"  Profitable campaigns: {benchmarks['count']}")
for name, q in QUANTILES.items():
    print(f"  {name.capitalize()} CPM (P{int(q*100)}): ${benchmarks[name]:.2f}")
print(f"  Mean CPM: ${benchmarks['mean']:.2f}")

print("\nTraining conversion classifier...")
clf, metrics, results = train_classifier(X, y_binary)

print(f"\n{'='*60}")
print(f"  V1: CONVERSION CLASSIFIER")
print(f"{'='*60}")
print(f"  Train: {metrics['train_size']}  Test: {metrics['test_size']}")
print(f"  CV F1:     {metrics['cv_f1']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
print(f"  Test Acc:  {metrics['accuracy']:.1%}  Prec: {metrics['precision']:.1%}  Recall: {metrics['recall']:.1%}  F1: {metrics['f1']:.4f}")

print(f"\n  Top Features:")
print(f"  {'-'*50}")
imp = get_feature_importance(clf, X)
for _, row in imp.head(10).iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

all_metrics = {"classifier": metrics, "benchmarks": benchmarks}
save_results(all_metrics, imp)
save_models(clf, benchmarks)
print(f"\nV1 models saved.")


# ══════════════════════════════════════════════════════════════
# V2: SIGNUP RATE MODEL + APPU WEIGHTS
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  V2: SIGNUP RATE + APPU PRICING MODEL")
print(f"{'='*60}")

# Load playbook data
print("\nLoading playbook data...")
df_playbook = load_playbook()
print(f"  Playbook rows: {len(df_playbook)}")

# Build APPU lookup
print("Building APPU lookup from platform demographics...")
appu_path = DATA_DIR / "purchase_demographics.csv"
ga_appu, tier_appu, global_appu = build_appu_lookup(appu_path)
print(f"  Global APPU: ${global_appu:.2f}")
print(f"  Tier APPUs: T1=${tier_appu[1]:.2f}  T2=${tier_appu[2]:.2f}  T3=${tier_appu[3]:.2f}")

# Build signup rate features
print("\nBuilding signup rate features...")
X_sr, y_sr, df_joined = build_signup_features(df_raw, df_playbook)
print(f"  Joined campaigns: {len(df_joined)}")
print(f"  Signup rate: median={np.expm1(y_sr).median():.4f} ({np.expm1(y_sr).median()*100:.2f}%)")

# Train signup rate model
print("\nTraining signup rate model...")
sr_model, sr_metrics = train_signup_model(X_sr, y_sr)

print(f"\n  Signup Rate Model:")
print(f"  Train: {sr_metrics['train_size']}  Test: {sr_metrics['test_size']}")
print(f"  CV R²:      {sr_metrics['cv_r2']:.4f} (+/- {sr_metrics['cv_r2_std']:.4f})")
print(f"  CV MAE:     {sr_metrics['cv_mae']:.6f} (+/- {sr_metrics['cv_mae_std']:.6f})")
print(f"  Test R²:    {sr_metrics['test_r2']:.4f}")
print(f"  Test MAE:   {sr_metrics['test_mae']:.6f}")

print(f"\n  Top Features:")
print(f"  {'-'*50}")
sr_imp = get_signup_feature_importance(sr_model)
for _, row in sr_imp.head(12).iterrows():
    if row["importance"] > 0.01:
        bar = "█" * int(row["importance"] * 80)
        print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

# Save V2 models
save_v2_models(sr_model, ga_appu, tier_appu, global_appu)
all_metrics["signup_model"] = sr_metrics
save_results(all_metrics, imp)
print(f"\nV2 models saved.")


# ══════════════════════════════════════════════════════════════
# BACKTEST: V2 vs V1
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  BACKTEST")
print(f"{'='*60}")

# Compute weighted APPU for each joined campaign
df_joined["w_appu"] = df_joined.apply(
    lambda r: compute_weighted_appu(r, ga_appu, tier_appu, global_appu), axis=1
)

# Predict signup rates
df_joined["pred_sr"] = predict_signup_rate(X_sr, sr_model)

# Project IAP and max price
df_joined["proj_iap"] = (
    df_joined["expected_views"] * df_joined["pred_sr"] * df_joined["w_appu"]
)
df_joined["max_price_v2"] = df_joined["proj_iap"] / PROFITABILITY_THRESHOLD
df_joined["profitable"] = (
    df_joined[IAP_COL] >= df_joined["campaign_cost_cleaned"] * PROFITABILITY_THRESHOLD
)

n_profitable = df_joined["profitable"].sum()
print(f"\n  Campaigns: {len(df_joined)}  Profitable: {n_profitable} ({df_joined['profitable'].mean()*100:.1f}%)")
print(f"\n  {'Model':<30s} {'Book':>5s} {'TP':>4s} {'FP':>4s} {'Miss':>5s} {'Prec':>6s} {'Rec':>6s} {'Spend':>11s} {'IAP':>11s} {'Net':>12s} {'ROI':>6s}")
print(f"  {'─'*105}")


def _show_backtest(label, max_prices, df_bt):
    wb = max_prices >= df_bt["campaign_cost_cleaned"]
    b = df_bt[wb]
    tp = (wb & df_bt["profitable"]).sum()
    fp = (wb & ~df_bt["profitable"]).sum()
    fn = (~wb & df_bt["profitable"]).sum()
    nb = wb.sum()
    pr = tp / nb if nb else 0
    rc = tp / df_bt["profitable"].sum()
    tc = b["campaign_cost_cleaned"].sum()
    ti = b[IAP_COL].sum()
    roi = ti / tc if tc else 0
    print(
        f"  {label:<30s} {nb:>5d} {tp:>4d} {fp:>4d} {fn:>5d}"
        f" {pr:>5.1%} {rc:>5.1%} ${tc:>9,.0f} ${ti:>9,.0f}"
        f" ${(ti - tc):>+10,.0f} {roi:>5.1%}"
    )


_show_backtest("V2: qual SR x APPU", df_joined["max_price_v2"], df_joined)

# Per APPU quartile
print(f"\n  Per APPU quartile:")
df_joined["aq"] = pd.qcut(
    df_joined["w_appu"], q=4,
    labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop",
)
for q, g in df_joined.groupby("aq", observed=True):
    wb = df_joined["max_price_v2"][g.index] >= g["campaign_cost_cleaned"]
    b = g[wb]
    if len(b) == 0:
        print(f"    {q}: 0 bookings")
        continue
    tc = b["campaign_cost_cleaned"].sum()
    ti = b[IAP_COL].sum()
    tp = (wb & g["profitable"]).sum()
    print(
        f"    {q:<10s}  appu=${g['w_appu'].median():.2f}"
        f"  book={wb.sum():>3d}/{len(g)}"
        f"  TP={tp:>2d}  spend=${tc:>8,.0f}  iap=${ti:>8,.0f}"
        f"  ROI={ti / tc:.1%}"
    )


# ══════════════════════════════════════════════════════════════
# SAMPLE V2 SCORECARDS
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  SAMPLE V2 SCORECARDS")
print(f"{'='*60}")

indices = np.random.RandomState(42).choice(len(df_joined), min(6, len(df_joined)), replace=False)
for i in indices:
    row = df_joined.iloc[i]
    name = row.get("clean_campaign_name", f"#{i}")
    ev = int(row["expected_views"])
    sr = float(row["pred_sr"])
    appu = float(row["w_appu"])
    cpm = float(row.get("expected_cpm", 0)) if pd.notna(row.get("expected_cpm")) else None

    card = format_scorecard_v2(name, ev, sr, appu, pitched_cpm=cpm)
    print(f"\n{card}")
