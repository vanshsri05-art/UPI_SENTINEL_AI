"""Evaluation helpers: metrics, thresholds, curves. Also a CLI to print the saved report.

    python -m ml.evaluate
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["Genuine", "Suspicious"]


def best_f1_threshold(y_true, proba) -> float:
    """Probability threshold that maximises F1 (chosen on validation data only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true, proba, threshold: float, full: bool = True) -> Dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }
    if full:
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        out["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
        out["classification_report"] = classification_report(
            y_true, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
    return out


def _downsample(*arrays, max_points: int = 120):
    n = len(arrays[0])
    if n <= max_points:
        return [np.asarray(a).round(5).tolist() for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return [np.asarray(a)[idx].round(5).tolist() for a in arrays]


def curve_points(y_true, proba) -> Dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fpr, tpr = _downsample(fpr, tpr)
    precision, recall = _downsample(precision, recall)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"precision": precision, "recall": recall}}


def main() -> None:
    from ml.config import METADATA_PATH

    if not METADATA_PATH.exists():
        raise SystemExit("No trained model found. Run:  python run.py train")
    meta = json.loads(METADATA_PATH.read_text())
    print(f"Dataset: {meta['dataset']['source']}  rows={meta['dataset']['rows']:,}")
    print(f"Selected model: {meta['selected_model']}  (decision threshold {meta['decision_threshold']:.3f})\n")
    header = f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'ROC-AUC':>9}{'PR-AUC':>8}"
    print("TEST-SET METRICS\n" + header + "\n" + "-" * len(header))
    for m in meta["models"]:
        t = m["test"]
        star = " *" if m["is_selected"] else ""
        print(f"{m['name']:<22}{t['accuracy']:>8.4f}{t['precision']:>8.4f}{t['recall']:>8.4f}"
              f"{t['f1']:>8.4f}{t['roc_auc']:>9.4f}{t['pr_auc']:>8.4f}{star}")
    sel = next(m for m in meta["models"] if m["is_selected"])
    cm = sel["test"]["confusion_matrix"]
    print(f"\nConfusion matrix of {sel['name']} (rows = actual, cols = predicted)")
    print(f"            Genuine  Suspicious\n  Genuine   {cm[0][0]:>7}  {cm[0][1]:>10}\n  Fraud     {cm[1][0]:>7}  {cm[1][1]:>10}")


if __name__ == "__main__":
    main()
