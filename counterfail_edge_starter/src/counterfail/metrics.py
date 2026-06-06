from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out.update({"precision": float(p), "recall": float(r)})
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    conf = np.maximum(y_prob, 1 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def risk_coverage(y_true, y_prob, coverages=(1.0, 0.9, 0.8, 0.7, 0.6)) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    conf = np.maximum(y_prob, 1 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    order = np.argsort(-conf)
    out = {}
    n = len(y_true)
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        idx = order[:k]
        risk = 1.0 - accuracy_score(y_true[idx], pred[idx])
        out[f"risk@cov{cov:.1f}"] = float(risk)
    return out
