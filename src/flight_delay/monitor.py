"""Monthly drift and decay check.

Two different questions get conflated under the word "drift", and this module answers
both because they have different remedies:

* **Input drift** — has the world changed shape? Measured with the Population Stability
  Index per feature. High PSI on `Origin` might just mean an airline restructured its
  network; it is a warning, not a verdict.

* **Performance decay** — is the model still right? BTS publishes the labels along with
  the flights, so unusually for a production system we can score last month's
  predictions against ground truth. This is the number that actually decides whether a
  retrain is needed; input drift without decay is noise, and decay without input drift
  still means the model is failing.

Run it after each new month lands. It exits non-zero when a threshold is breached, so
the scheduled workflow can open an issue and trigger a retrain rather than a human
noticing months later.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from flight_delay.features import xy
from flight_delay.metrics import evaluate
from flight_delay.schema import NUMERIC_FEATURES, TARGET

log = logging.getLogger(__name__)

# Conventional PSI reading: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant.
PSI_WARN = 0.10
PSI_ALERT = 0.25


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature.

    Bin edges come from the reference quantiles so the reference is uniform by
    construction and any imbalance in `current` is real movement rather than an
    artifact of where the edges fell.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < bins or len(cur) < bins:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    # Floor at a small epsilon: an empty bin would otherwise make PSI infinite and
    # a single missing category would drown out every real signal.
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """PSI for every numeric feature, plus the label rate."""
    scores = {
        feature: psi(reference[feature], current[feature])
        for feature in NUMERIC_FEATURES
        if feature in reference.columns and feature in current.columns
    }
    scores["__label_rate_shift"] = float(current[TARGET].mean() - reference[TARGET].mean())
    return scores


def score_current(model_dir: Path, current: pd.DataFrame) -> dict[str, float]:
    """Evaluate the deployed model against last month's ground truth."""
    model = mlflow.sklearn.load_model(str(model_dir))
    X, y = xy(current)
    return evaluate(y, model.predict_proba(X)[:, 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True, help="parquet the model trained on")
    p.add_argument("--current", type=Path, required=True, help="parquet for the new month")
    p.add_argument("--model-dir", type=Path, default=Path("models/candidate"))
    p.add_argument("--baseline-metrics", type=Path, default=Path("models/candidate_metrics.json"))
    p.add_argument("--max-pr-auc-decay", type=float, default=0.02)
    p.add_argument("--out", type=Path, default=Path("models/drift_report.json"))
    p.add_argument(
        "--strict",
        action="store_true",
        help="also fail on input drift alone; off by default, see module docstring",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    reference = pd.read_parquet(args.reference)
    current = pd.read_parquet(args.current)
    psi_scores = compare(reference, current)
    live = score_current(args.model_dir, current)

    # What counts as an alert, and what is merely worth reporting.
    #
    # Input drift alone does not trigger a retrain. Running this January-to-April
    # produces PSI above 3.0 on temperature — an enormous number — while live PR-AUC
    # is unchanged to four decimal places, because the seasons changing is exactly the
    # variation the model was trained across. Wiring "PSI > 0.25" straight to a retrain
    # is the most common way drift monitoring becomes an expensive noise generator.
    #
    # Performance decay against ground truth is the decisive signal. Input drift is
    # recorded as context, so that when decay does appear there is already an answer to
    # "what changed?" waiting in the report.
    warnings: list[str] = []
    alerts: list[str] = []

    drifted = {k: v for k, v in psi_scores.items() if k != "__label_rate_shift" and v >= PSI_ALERT}
    if drifted:
        warnings.append(
            "Input drift (PSI >= 0.25): "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(drifted.items(), key=lambda kv: -kv[1]))
        )

    decay = None
    if args.baseline_metrics.exists():
        at_training = json.loads(args.baseline_metrics.read_text())["model"]["pr_auc"]
        decay = at_training - live["pr_auc"]
        if decay > args.max_pr_auc_decay:
            alerts.append(
                f"Performance decay: PR-AUC {live['pr_auc']:.4f} on new data versus "
                f"{at_training:.4f} at training ({-decay:+.4f}). "
                + (
                    f"Co-occurring input drift: {'; '.join(warnings)}"
                    if warnings
                    else "No significant input drift — investigate label or upstream changes."
                )
            )
    else:
        warnings.append("No baseline metrics on record; decay could not be evaluated.")

    if args.strict and warnings and not alerts:
        alerts.extend(warnings)

    report = {
        "reference": str(args.reference),
        "current": str(args.current),
        "rows_scored": int(len(current)),
        "psi": psi_scores,
        "live_metrics": live,
        "pr_auc_decay": decay,
        "warnings": warnings,
        "alerts": alerts,
        "retrain_recommended": bool(alerts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"Drift report  {args.reference.name} -> {args.current.name}")
    print("=" * 62)
    for feature, value in sorted(psi_scores.items(), key=lambda kv: -(kv[1] or 0)):
        if feature.startswith("__"):
            continue
        flag = "ALERT" if value >= PSI_ALERT else "warn " if value >= PSI_WARN else "     "
        print(f"  {flag}  {feature:<26} PSI {value:.4f}")
    print("-" * 62)
    print(f"  label rate shift {psi_scores['__label_rate_shift']:+.4f}")
    print(f"  live PR-AUC      {live['pr_auc']:.4f}" + (f"  (decay {decay:+.4f})" if decay else ""))
    print("=" * 62)

    for warning in warnings:
        print(f"  note   {warning}")
    if alerts:
        for alert in alerts:
            print(f"ALERT  {alert}", file=sys.stderr)
        return 1
    print(
        "  No retrain needed: performance is holding"
        + (" despite the input drift above." if warnings else " and inputs are stable.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
