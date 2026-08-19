"""Evaluation metrics chosen for an imbalanced, probability-output problem.

Roughly 19% of flights arrive late, so accuracy is actively misleading: predicting
"on time" for everything scores 81%. Three families of metric are reported instead,
because they answer three different questions:

* **Ranking** (PR-AUC, ROC-AUC) — can the model order flights by risk? PR-AUC is the
  primary gate metric since it focuses on the minority class we care about.
* **Calibration** (Brier, ECE) — when the model says "35% chance of delay", is it right
  35% of the time? A ranking-only model is fine for triage but wrong for anything a
  traveller reads as a probability, which is exactly what the demo UI shows.
* **Business framing** (lift in the top decile) — of the 10% of flights the model flags
  as riskiest, how many actually run late, versus the base rate? This is the number a
  non-technical stakeholder can act on.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Mean gap between predicted probability and observed frequency, weighted by bin size."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(error)


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray, quantile: float = 0.9) -> float:
    """Delay rate among the riskiest flights, divided by the overall delay rate."""
    base = y_true.mean()
    if base == 0:
        return float("nan")
    threshold = np.quantile(y_prob, quantile)
    flagged = y_prob >= threshold
    if not flagged.any():
        return float("nan")
    return float(y_true[flagged].mean() / base)


def evaluate(y_true, y_prob) -> dict[str, float]:
    """Compute the full metric set for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "top_decile_lift": top_decile_lift(y_true, y_prob),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
