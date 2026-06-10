"""CounterFail-Edge evaluation metrics.

Label convention:
  - y_true = 1  →  success
  - y_true = 0  →  failure
  - y_pred = 1  if  y_prob >= threshold  (predicted success)

Confusion matrix naming:
  tn_failure_correct     = true failure predicted failure
  fp_failure_as_success  = true failure predicted success
  fn_success_as_failure  = true success predicted failure
  tp_success_correct     = true success predicted success
"""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Compute explicit success/failure metrics.  No bare 'f1'/'precision'/'recall'."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix: labels=[0,1] so row-0=failure, row-1=success
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    n_total = len(y_true)
    pred_success = int(y_pred.sum())
    pred_failure = n_total - pred_success

    out: dict[str, float] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "success_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "success_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "success_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "failure_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "failure_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "failure_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "threshold": float(threshold),
        "pred_success_rate": float(pred_success / max(n_total, 1)),
        "pred_failure_rate": float(pred_failure / max(n_total, 1)),
        # Confusion matrix
        "tn_failure_correct": int(tn),
        "fp_failure_as_success": int(fp),
        "fn_success_as_failure": int(fn),
        "tp_success_correct": int(tp),
    }

    # AUC metrics
    try:
        out["auroc_success"] = float(roc_auc_score(y_true, y_prob))
        out["auprc_success"] = float(average_precision_score(y_true, y_prob))
        out["auroc_failure"] = float(roc_auc_score(1 - y_true, 1 - y_prob))
        out["auprc_failure"] = float(average_precision_score(1 - y_true, 1 - y_prob))
    except ValueError:
        out["auroc_success"] = float("nan")
        out["auprc_success"] = float("nan")
        out["auroc_failure"] = float("nan")
        out["auprc_failure"] = float("nan")

    return out


def per_type_recall(y_true, y_prob, failure_types, threshold: float = 0.5) -> dict:
    """Per failure-type recall.

    For each failure_type:
      - 'success': recall = fraction correctly predicted as success (y_pred=1)
      - any failure type: recall = fraction correctly predicted as failure (y_pred=0)

    Returns dict mapping type -> {"n": int, "recall": float}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    result = {}
    for ft in sorted(set(failure_types)):
        idx = np.array([x == ft for x in failure_types])
        n = int(idx.sum())
        if n == 0:
            continue
        if ft == "success":
            recall = float((y_pred[idx] == 1).mean())
        else:
            recall = float((y_pred[idx] == 0).mean())
        result[ft] = {"n": n, "recall": recall}
    return result


def per_type_mean_recall(y_true, y_prob, failure_types, threshold: float = 0.5) -> float:
    """Mean recall over all available types (including success and each failure type).

    This is important because balanced_acc can hide poor wrong-object/wrong-placement performance.
    """
    pt = per_type_recall(y_true, y_prob, failure_types, threshold)
    if not pt:
        return 0.0
    recalls = [v["recall"] for v in pt.values()]
    return float(np.mean(recalls))


def group_hmean_recall(y_true, y_prob, failure_types, threshold: float = 0.5) -> float:
    """Harmonic mean of success_group_recall and failure_type_mean_recall.

    This metric prevents checkpoint collapse: an all-failure predictor gets ~0.0
    because success_recall ≈ 0, and an all-success predictor also gets ~0.0
    because failure_type_mean_recall ≈ 0.
    """
    pt = per_type_recall(y_true, y_prob, failure_types, threshold)
    if not pt:
        return 0.0
    success_recall = pt.get("success", {}).get("recall", 0.0)
    failure_recalls = [v["recall"] for k, v in pt.items() if k != "success"]
    if not failure_recalls:
        return 0.0
    failure_mean_recall = float(np.mean(failure_recalls))
    denom = success_recall + failure_mean_recall
    if denom == 0:
        return 0.0
    return float(2.0 * success_recall * failure_mean_recall / denom)


def group_balanced_type_recall(y_true, y_prob, failure_types, threshold: float = 0.5) -> float:
    """0.5 * success_recall + 0.5 * mean(failure_type_recalls).

    Arithmetic-mean variant of group_hmean_recall. Less strict than harmonic mean
    but still prevents collapse.
    """
    pt = per_type_recall(y_true, y_prob, failure_types, threshold)
    if not pt:
        return 0.0
    success_recall = pt.get("success", {}).get("recall", 0.0)
    failure_recalls = [v["recall"] for k, v in pt.items() if k != "success"]
    if not failure_recalls:
        return success_recall * 0.5
    return float(0.5 * success_recall + 0.5 * np.mean(failure_recalls))


# Metrics that require failure_types for threshold sweep
_TYPE_AWARE_METRICS = frozenset({
    "per_type_mean_recall", "group_hmean_recall", "group_balanced_type_recall",
})


def find_best_threshold(
    y_true,
    y_prob,
    metric: str = "macro_f1",
    lo: float = 0.05,
    hi: float = 0.95,
    steps: int = 181,
    failure_types=None,
) -> dict[str, float]:
    """Sweep thresholds and return metrics at the best one.

    Supported metrics:
      macro_f1, balanced_acc, failure_f1, failure_recall, success_f1,
      per_type_mean_recall, group_hmean_recall, group_balanced_type_recall
      (the last three require failure_types).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(lo, hi, steps)

    _type_metric_fns = {
        "per_type_mean_recall": per_type_mean_recall,
        "group_hmean_recall": group_hmean_recall,
        "group_balanced_type_recall": group_balanced_type_recall,
    }

    best: Optional[tuple] = None
    for threshold in thresholds:
        th = float(threshold)
        if metric in _TYPE_AWARE_METRICS:
            if failure_types is None:
                raise ValueError(f"failure_types required for {metric} metric")
            score = _type_metric_fns[metric](y_true, y_prob, failure_types, th)
        else:
            row = binary_metrics(y_true, y_prob, threshold=th)
            score = row.get(metric)
            if score is None:
                raise KeyError(f"Unknown threshold metric: {metric}")
        if best is None or score > best[0]:
            best = (score, th)

    # Re-compute full metrics at best threshold
    best_th = best[1]
    metrics = binary_metrics(y_true, y_prob, threshold=best_th)
    if failure_types is not None:
        metrics["per_type_mean_recall"] = per_type_mean_recall(y_true, y_prob, failure_types, best_th)
        metrics["group_hmean_recall"] = group_hmean_recall(y_true, y_prob, failure_types, best_th)
        metrics["group_balanced_type_recall"] = group_balanced_type_recall(y_true, y_prob, failure_types, best_th)
    return metrics


def expected_calibration_error(y_true, y_prob, threshold: float = 0.5, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    When threshold=0.5 this is standard probability ECE.
    When threshold != 0.5, confidence is computed as margin from the decision
    boundary: conf = abs(p - threshold), giving a 'decision ECE'.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if abs(threshold - 0.5) < 1e-6:
        # Standard probability ECE
        conf = np.maximum(y_prob, 1 - y_prob)
        pred = (y_prob >= 0.5).astype(int)
        bins = np.linspace(0.5, 1.0, n_bins + 1)
    else:
        # Decision ECE: confidence = distance from threshold
        conf = np.abs(y_prob - threshold)
        pred = (y_prob >= threshold).astype(int)
        bins = np.linspace(0.0, max(threshold, 1.0 - threshold), n_bins + 1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo_b) & (conf < hi_b)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def risk_coverage(
    y_true,
    y_prob,
    coverages=(1.0, 0.9, 0.8, 0.7, 0.6),
    threshold: float = 0.5,
    confidence_mode: str = "margin",
) -> dict[str, float]:
    """Risk at various coverage levels.

    confidence_mode:
      - "margin": conf = abs(y_prob - threshold)   (meaningful for non-0.5 thresholds)
      - "prob":   conf = max(y_prob, 1 - y_prob)    (original, assumes threshold=0.5)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if confidence_mode == "margin":
        conf = np.abs(y_prob - threshold)
    else:
        conf = np.maximum(y_prob, 1 - y_prob)

    pred = (y_prob >= threshold).astype(int)
    order = np.argsort(-conf)
    out = {}
    n = len(y_true)
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        idx = order[:k]
        risk = 1.0 - accuracy_score(y_true[idx], pred[idx])
        out[f"risk@cov{cov:.1f}"] = float(risk)
    return out
