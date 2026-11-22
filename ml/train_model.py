"""Train, compare and save the fraud model.

    python -m ml.train_model                       # synthetic UPI-style data (default)
    python -m ml.train_model --source paysim       # PaySim CSV placed at dataset/paysim.csv
    python -m ml.train_model --quick               # fewer trees, for a fast smoke run

Pipeline: load -> clean -> stratified 70/15/15 split -> fit 4 models on TRAIN only ->
choose the decision threshold on VALIDATION -> pick the model with the best validation F1 ->
report everything on the untouched TEST set. Class imbalance is handled with class weights
(applied inside the training data only), so no synthetic rows are ever created from test data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml import config  # noqa: E402
from ml.anomaly_detection import (  # noqa: E402
    compare_with_supervised, fit_anomaly_model, save_anomaly_model, score_anomalies)
from ml.evaluate import best_f1_threshold, compute_metrics, curve_points  # noqa: E402
from ml.explainability import SHAP_AVAILABLE, compute_global_importance  # noqa: E402
from ml.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS, TARGET_COLUMN, build_features, clean_dataset, load_dataset, split_data)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

RS = config.RANDOM_STATE


def build_candidates(pos_weight: float, quick: bool = False) -> dict:
    n_trees = 80 if quick else 300
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=RS),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_trees, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RS),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=n_trees, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RS)
    return models


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train UPI Sentinel AI fraud models")
    ap.add_argument("--source", choices=["synthetic", "paysim"], default="synthetic")
    ap.add_argument("--max-rows", type=int, default=200_000, help="PaySim only: row cap (all fraud kept)")
    ap.add_argument("--quick", action="store_true", help="fewer trees for a fast run")
    args = ap.parse_args(argv)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/6] Loading '{args.source}' dataset ...")
    raw = load_dataset(args.source, max_rows=args.max_rows)
    df, report = clean_dataset(raw)
    print(f"      cleaning report: {report}")

    train, val, test = split_data(df)
    X_tr, y_tr = build_features(train), train[TARGET_COLUMN]
    X_va, y_va = build_features(val), val[TARGET_COLUMN]
    X_te, y_te = build_features(test), test[TARGET_COLUMN]
    assert list(X_tr.columns) == FEATURE_COLUMNS
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"      train/val/test = {len(train):,}/{len(val):,}/{len(test):,}   "
          f"fraud rate = {df[TARGET_COLUMN].mean():.2%}   class weight ratio = {pos_weight:.1f}")

    if not XGBOOST_AVAILABLE:
        print("      NOTE: xgboost is not installed - XGBoost is skipped (pip install xgboost).")

    print("[2/6] Training and comparing models ...")
    results, fitted = [], {}
    for name, model in build_candidates(pos_weight, args.quick).items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        secs = time.time() - t0
        va_proba = model.predict_proba(X_va)[:, 1]
        thr = best_f1_threshold(y_va, va_proba)
        te_proba = model.predict_proba(X_te)[:, 1]
        entry = {
            "name": name,
            "threshold": thr,
            "train_seconds": round(secs, 2),
            "validation": compute_metrics(y_va, va_proba, thr, full=False),
            "test": compute_metrics(y_te, te_proba, thr, full=True),
            "curves": curve_points(y_te, te_proba),
            "is_selected": False,
        }
        results.append(entry)
        fitted[name] = model
        t = entry["test"]
        print(f"      {name:<20} val-F1={entry['validation']['f1']:.3f} | test: P={t['precision']:.3f} "
              f"R={t['recall']:.3f} F1={t['f1']:.3f} AUC={t['roc_auc']:.3f} ({secs:.1f}s)")

    print("[3/6] Selecting final model (best validation F1, ties -> higher recall) ...")
    best = max(results, key=lambda r: (round(r["validation"]["f1"], 4), r["validation"]["recall"]))
    best["is_selected"] = True
    model = fitted[best["name"]]
    print(f"      selected: {best['name']}  (decision threshold = {best['threshold']:.3f})")
    joblib.dump(model, config.MODEL_PATH, compress=3)

    print("[4/6] Computing global feature importance ...")
    importance = compute_global_importance(model, X_te, y_te)
    print(f"      method: {importance['method']}")

    print("[5/6] Fitting Isolation Forest (unsupervised) and comparing with the classifier ...")
    bundle = fit_anomaly_model(X_tr)
    save_anomaly_model(bundle)
    a_flags, _ = score_anomalies(bundle, X_te)
    sup_flags = model.predict_proba(X_te)[:, 1] >= best["threshold"]
    anomaly_report = compare_with_supervised(y_te, sup_flags, a_flags)

    print("[6/6] Saving metadata ...")
    metadata = {
        "project": "UPI Sentinel AI",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "xgboost_available": XGBOOST_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
        "dataset": {
            "source": args.source,
            "rows": int(len(df)),
            "fraud_rate": float(df[TARGET_COLUMN].mean()),
            "cleaning_report": report,
            "note": ("PaySim is synthetic MOBILE-MONEY data, not real UPI data." if args.source == "paysim"
                     else "Synthetic UPI-style data produced by dataset/generate_sample_data.py - "
                          "not real UPI transactions."),
        },
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "feature_columns": FEATURE_COLUMNS,
        "baseline": {k: float(v) for k, v in X_tr.median().items()},
        "selected_model": best["name"],
        "decision_threshold": best["threshold"],
        "selection_criterion": "Highest validation F1 (balances precision and recall); "
                               "threshold tuned on validation; metrics reported on held-out test set.",
        "imbalance_handling": "Class weights / scale_pos_weight (training data only).",
        "models": results,
        "global_importance": importance,
        "anomaly": {**anomaly_report, "contamination": bundle["contamination"]},
        "risk_thresholds": {"low_max": config.RISK_LOW_MAX, "medium_max": config.RISK_MEDIUM_MAX,
                            "note": "Project-defined thresholds, not an official banking standard."},
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved: {config.MODEL_PATH.name}, {config.ANOMALY_MODEL_PATH.name}, {config.METADATA_PATH.name} "
          f"in {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
