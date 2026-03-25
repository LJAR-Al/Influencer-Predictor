"""Two-stage model: classifier + profitable CPM benchmarks for pricing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.ensemble import GradientBoostingClassifier
import json
from pathlib import Path

from src.config import (
    QUANTILES, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    PROFITABILITY_THRESHOLD, IAP_COL,
)


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


def compute_profitable_benchmarks(df_model):
    """
    Compute CPM benchmarks from historically profitable campaigns.
    Profitable = d7 users' IAP at day 14 >= 10% of cost paid.
    """
    cost = pd.to_numeric(df_model.get("campaign_cost_cleaned", 0), errors="coerce")
    rev = df_model[IAP_COL]
    ev = pd.to_numeric(df_model.get("expected_views", 0), errors="coerce")
    roi = rev / cost

    mask = (roi >= PROFITABILITY_THRESHOLD) & (cost > 0) & (ev > 0)
    prof = df_model[mask].copy()
    prof["paid_cpm"] = cost[prof.index] / (ev[prof.index] / 1000)

    benchmarks = {}
    for name, q in QUANTILES.items():
        benchmarks[name] = float(prof["paid_cpm"].quantile(q))

    benchmarks["mean"] = float(prof["paid_cpm"].mean())
    benchmarks["count"] = len(prof)

    return benchmarks


def train_classifier(X, y_binary):
    """Train the conversion classifier and return it with metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    num_feats, cat_feats = _get_feature_lists(X_train)

    clf = _build_classifier(num_feats, cat_feats)
    clf_cv = cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="f1")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    conv_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_f1": float(clf_cv.mean()),
        "cv_f1_std": float(clf_cv.std()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    results = {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "conv_prob": conv_prob,
    }

    return clf, metrics, results


def get_feature_importance(pipeline, X):
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
