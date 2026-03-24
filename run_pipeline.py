"""Run the influencer pricing pipeline (two-stage + quantile)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from src.data import load_raw, clean, build_features
from src.model import train_two_stage_quantile, get_feature_importance, save_results
from src.predict import save_models, format_scorecard
from src.config import QUANTILES, MIN_MARGIN

# 1. Load and clean
print("Loading data...")
df_raw = load_raw()
df = clean(df_raw)
print(f"  Raw rows: {len(df_raw)}")

# 2. Build features
print("Building features...")
X, y_log, y_binary, df_model = build_features(df)
print(f"  Modeling rows:  {len(X)}")
print(f"  Features:       {list(X.columns)}")
print(f"  Converters:     {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
print(f"  Non-converters: {(y_binary == 0).sum()} ({(1 - y_binary.mean())*100:.1f}%)")
rev_real = np.expm1(y_log)
print(f"  IAP Revenue:    mean=${rev_real.mean():,.0f}  median=${rev_real.median():,.0f}")

# 3. Train
print("\nTraining two-stage quantile model...")
models, metrics, results = train_two_stage_quantile(X, y_log, y_binary)

# 4. Results
cm = metrics["classifier"]
rm = metrics["regressor"]

print(f"\n{'='*60}")
print(f"  STAGE 1: CONVERSION CLASSIFIER")
print(f"{'='*60}")
print(f"  CV F1:     {cm['cv_f1']:.4f} (+/- {cm['cv_f1_std']:.4f})")
print(f"  Test Acc:  {cm['accuracy']:.1%}  Prec: {cm['precision']:.1%}  Recall: {cm['recall']:.1%}  F1: {cm['f1']:.4f}")

print(f"\n{'='*60}")
print(f"  STAGE 2: IAP REVENUE PREDICTION (converters)")
print(f"{'='*60}")
print(f"  Converters in train:  {rm['converters_train']}")
print(f"  L2 model — MAE: ${rm['l2_mae']:,.2f}  RMSE: ${rm['l2_rmse']:,.2f}  R²: {rm['l2_r2']:.4f}")
for name in QUANTILES:
    print(f"  {name.capitalize():<15} MAE: ${rm[f'{name}_mae']:,.2f}")

print(f"\n{'='*60}")
print(f"  PRICING RELIABILITY")
print(f"{'='*60}")
print(f"  Price range coverage (actual within cons-aggr):  {rm['price_range_coverage']:.1%}")
print(f"  Conservative safety (actual >= conservative):    {rm['conservative_safety_rate']:.1%}")
print(f"  Conservative profit rate (actual > cons price):  {rm['conservative_profit_rate']:.1%}")

# 5. Feature importance
print(f"\n  Classifier — Top Features:")
print(f"  {'-'*50}")
clf_imp = get_feature_importance(models["classifier"], X)
for _, row in clf_imp.head(8).iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

# Use converter subset for regressor importance
conv_X = X[y_binary == 1]
print(f"\n  Revenue Model — Top Features:")
print(f"  {'-'*50}")
reg_imp = get_feature_importance(models["l2"], conv_X)
for _, row in reg_imp.head(8).iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

# 6. Save
save_results(metrics, clf_imp)
reg_imp.to_csv("results/feature_importance_regressor.csv", index=False)
save_models(models)
print(f"\nModels and results saved.")

# 7. Sample scorecards
print(f"\n{'='*60}")
print(f"  SAMPLE CREATOR SCORECARDS")
print(f"{'='*60}")

X_test = results["X_test"]
y_test_real = results["y_test_real"]

indices = np.random.RandomState(42).choice(len(X_test), min(6, len(X_test)), replace=False)
for i in indices:
    row_X = X_test.iloc[[i]]
    ev = int(row_X["expected_views"].values[0])
    actual_rev = y_test_real.iloc[i]
    orig_idx = X_test.index[i]
    creator = df_model.loc[orig_idx, "creator_id"] if "creator_id" in df_model.columns else f"#{i}"

    card = format_scorecard(row_X, ev, models, creator_name=creator)
    print(f"\n{card}")
    print(f"  Actual IAP revenue: ${actual_rev:,.2f}")
    if actual_rev > 0:
        actual_max = actual_rev / MIN_MARGIN
        print(f"  Actual max price (@{(MIN_MARGIN-1)*100:.0f}% margin): ${actual_max:,.2f}")
