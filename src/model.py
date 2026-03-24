"""Two-stage quantile model for influencer pricing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import json
from pathlib import Path

from src.config import QUANTILES, RANDOM_STATE, TEST_SIZE, CV_FOLDS, MIN_MARGIN


def _get_feature_lists(X):
    num_feats = list(X.select_dtypes(include=[np.number]).columns)
    cat_feats = list(X.select_dtypes(exclude=[np.number]).columns)
    return num_feats, cat_feats


def _build_preprocessor(num_feats, cat_feats):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ],
        remainder="drop",
    )


def _build_classifier(num_feats, cat_feats):
    return Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats)),
        ("model", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )),
    ])


def _build_quantile_regressor(num_feats, cat_feats, quantile=0.5):
    return Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats)),
        ("model", GradientBoostingRegressor(
            loss="quantile", alpha=quantile,
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )),
    ])


def _build_l2_regressor(num_feats, cat_feats):
    return Pipeline([
        ("preprocessor", _build_preprocessor(num_feats, cat_feats)),
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )),
    ])


def train_two_stage_quantile(X, y_log, y_binary):
    """
    Two-stage pricing model:
      Stage 1 — Classify: will this creator's campaign convert?
      Stage 2 — Quantile regression on converters: predict IAP revenue ranges.

    Returns trained models, metrics, and test results.
    """
    X_train, X_test, y_log_train, y_log_test, y_bin_train, y_bin_test = train_test_split(
        X, y_log, y_binary,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    num_feats, cat_feats = _get_feature_lists(X_train)

    # === Stage 1: Conversion classifier ===
    print("  Training conversion classifier...")
    clf = _build_classifier(num_feats, cat_feats)
    clf_cv = cross_val_score(clf, X_train, y_bin_train, cv=CV_FOLDS, scoring="f1")
    clf.fit(X_train, y_bin_train)

    y_bin_pred = clf.predict(X_test)
    conv_prob = clf.predict_proba(X_test)[:, 1]

    clf_metrics = {
        "cv_f1": float(clf_cv.mean()),
        "cv_f1_std": float(clf_cv.std()),
        "accuracy": float(accuracy_score(y_bin_test, y_bin_pred)),
        "precision": float(precision_score(y_bin_test, y_bin_pred, zero_division=0)),
        "recall": float(recall_score(y_bin_test, y_bin_pred, zero_division=0)),
        "f1": float(f1_score(y_bin_test, y_bin_pred, zero_division=0)),
    }

    # === Stage 2: Quantile regression on converters only ===
    conv_mask_train = y_bin_train == 1
    X_train_conv = X_train[conv_mask_train]
    y_log_train_conv = y_log_train[conv_mask_train]

    quantile_models = {}
    quantile_preds_log = {}

    for name, q in QUANTILES.items():
        print(f"  Training {name} quantile model (q={q})...")
        pipe = _build_quantile_regressor(num_feats, cat_feats, quantile=q)
        pipe.fit(X_train_conv, y_log_train_conv)
        quantile_models[name] = pipe
        quantile_preds_log[name] = pipe.predict(X_test)

    # L2 model for standard metrics
    print("  Training L2 regressor for evaluation...")
    l2 = _build_l2_regressor(num_feats, cat_feats)
    l2_cv = cross_val_score(
        l2, X_train_conv, y_log_train_conv,
        cv=min(CV_FOLDS, max(2, len(X_train_conv) // 10)),
        scoring="neg_mean_absolute_error",
    )
    l2.fit(X_train_conv, y_log_train_conv)
    quantile_models["l2"] = l2

    # === Combined predictions ===
    # For each quantile: zero out if P(conversion) < 50%, else use quantile prediction
    y_test_real = np.expm1(y_log_test)
    combined_preds = {}
    for name in QUANTILES:
        log_pred = quantile_preds_log[name]
        # Zero prediction for predicted non-converters
        combined_log = np.where(conv_prob >= 0.5, log_pred, 0.0)
        combined_preds[name] = np.expm1(combined_log)

    # L2 combined
    l2_log_pred = l2.predict(X_test)
    l2_combined = np.expm1(np.where(conv_prob >= 0.5, l2_log_pred, 0.0))

    # === Metrics ===
    reg_metrics = {
        "converters_train": int(conv_mask_train.sum()),
        "l2_cv_mae_log": float(-l2_cv.mean()),
        "l2_mae": float(mean_absolute_error(y_test_real, l2_combined)),
        "l2_rmse": float(np.sqrt(mean_squared_error(y_test_real, l2_combined))),
        "l2_r2": float(r2_score(y_test_real, l2_combined)),
    }

    for name in QUANTILES:
        reg_metrics[f"{name}_mae"] = float(mean_absolute_error(y_test_real, combined_preds[name]))

    # Price range coverage
    cons = combined_preds["conservative"]
    aggr = combined_preds["aggressive"]
    in_range = ((y_test_real >= cons) & (y_test_real <= aggr)).mean()
    above_cons = (y_test_real >= cons).mean()
    reg_metrics["price_range_coverage"] = float(in_range)
    reg_metrics["conservative_safety_rate"] = float(above_cons)

    # Margin safety: at conservative pricing, what % would be profitable?
    # If we set price = conservative_revenue / 1.1, how often is actual_revenue > price?
    cons_price = cons / MIN_MARGIN
    profitable = (y_test_real > cons_price).mean()
    reg_metrics["conservative_profit_rate"] = float(profitable)

    all_metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classifier": clf_metrics,
        "regressor": reg_metrics,
    }

    results = {
        "X_test": X_test,
        "y_log_test": y_log_test,
        "y_bin_test": y_bin_test,
        "y_test_real": y_test_real,
        "conv_prob": conv_prob,
        "combined_preds": combined_preds,
        "l2_combined": l2_combined,
    }

    all_models = {"classifier": clf, **quantile_models}
    return all_models, all_metrics, results


def get_feature_importance(pipeline, X):
    """Extract feature importances from a trained pipeline."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    num_feats, cat_feats = _get_feature_lists(X)
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_feats))
    all_feature_names = num_feats + cat_feature_names
    importances = model.feature_importances_
    return pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)


def save_results(metrics, importance_df, output_dir="results"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    importance_df.to_csv(out / "feature_importance.csv", index=False)
