"""Run the influencer pricing pipeline (classifier + profitable CPM benchmarks)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from src.data import load_raw, clean, build_features
from src.model import train_classifier, compute_profitable_benchmarks, get_feature_importance, save_results
from src.predict import save_models, format_scorecard
from src.config import QUANTILES, PROFITABILITY_THRESHOLD

# 1. Load and clean
print("Loading data...")
df_raw = load_raw()
df = clean(df_raw)
print(f"  Raw rows: {len(df_raw)}")

# 2. Build features
print("Building features...")
X, y_log, y_binary, df_model = build_features(df)
print(f"  Modeling rows:  {len(X)}")
print(f"  Converters:     {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
print(f"  Non-converters: {(y_binary == 0).sum()} ({(1 - y_binary.mean())*100:.1f}%)")

# 3. Compute profitable CPM benchmarks
print("\nComputing profitable campaign benchmarks...")
benchmarks = compute_profitable_benchmarks(df_model)
print(f"  Profitable campaigns: {benchmarks['count']}")
for name, q in QUANTILES.items():
    print(f"  {name.capitalize()} CPM (P{int(q*100)}): ${benchmarks[name]:.2f}")
print(f"  Mean CPM: ${benchmarks['mean']:.2f}")

# 4. Train classifier
print("\nTraining conversion classifier...")
clf, metrics, results = train_classifier(X, y_binary)

print(f"\n{'='*60}")
print(f"  CONVERSION CLASSIFIER")
print(f"{'='*60}")
print(f"  Train: {metrics['train_size']}  Test: {metrics['test_size']}")
print(f"  CV F1:     {metrics['cv_f1']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
print(f"  Test Acc:  {metrics['accuracy']:.1%}  Prec: {metrics['precision']:.1%}  Recall: {metrics['recall']:.1%}  F1: {metrics['f1']:.4f}")

# 5. Feature importance
print(f"\n  Top Features:")
print(f"  {'-'*50}")
imp = get_feature_importance(clf, X)
for _, row in imp.head(10).iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:<40} {row['importance']:.4f} {bar}")

# 6. Save
all_metrics = {"classifier": metrics, "benchmarks": benchmarks}
save_results(all_metrics, imp)
save_models(clf, benchmarks)
print(f"\nModels and benchmarks saved.")

# 7. Sample scorecards
print(f"\n{'='*60}")
print(f"  SAMPLE SCORECARDS")
print(f"{'='*60}")

X_test = results["X_test"]
conv_prob = results["conv_prob"]

indices = np.random.RandomState(42).choice(len(X_test), min(6, len(X_test)), replace=False)
for i in indices:
    row_X = X_test.iloc[[i]]
    ev = int(row_X["expected_views"].values[0])
    orig_idx = X_test.index[i]
    creator = df_model.loc[orig_idx, "creator_id"] if "creator_id" in df_model.columns else f"#{i}"

    card = format_scorecard(row_X, ev, clf, benchmarks, creator_name=creator)
    print(f"\n{card}")
