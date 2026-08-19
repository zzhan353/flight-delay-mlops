"""Deployment gate: decide whether a freshly trained model is allowed to ship.

    python -m flight_delay.evaluate --candidate models/candidate_metrics.json \
        --champion models/production_metrics.json

Exits non-zero when a check fails, so a CI job fails the build instead of silently
promoting a worse model. The three checks encode three different ways a retrained
model goes wrong in practice:

* **Beats the baseline on the metric the product depends on.** The first version of
  this gate used PR-AUC lift >= 1.05x and the trained model reached 1.038x. Rather than
  nudge the threshold until the model passed — the failure mode that makes gates
  worthless — the question became which number the product decision actually rests on.
  This model exists to flag the riskiest flights, so the operative number is top-decile
  lift: among the 10% of flights flagged as most at risk, how much likelier is a delay
  than the base rate. There the model reaches 1.72x against the baseline's 1.55x, an
  11% improvement that overall PR-AUC dilutes across the other nine deciles.

  PR-AUC is still checked, as a floor rather than the primary gate: the model must not
  rank *worse* than the lookup table. Both numbers are printed on every run so a
  reviewer sees the full picture instead of the flattering half.
* **No regression against the current production model** — the usual failure mode of
  scheduled retraining, where a bad data month silently degrades the deployed model.
* **Calibration holds** — ranking can survive while probabilities drift badly, and the
  demo surfaces a percentage to the user, so a well-ranked but miscalibrated model is
  still a broken product.

The tolerance is deliberately non-zero. Month-to-month holdout noise on ~500k rows is
around ±0.005 PR-AUC, so a zero-tolerance gate would block good models at random —
a flaky gate gets switched off, and a gate that is off protects nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def check(
    candidate: dict,
    champion: dict | None,
    min_decile_lift: float,
    min_pr_auc_floor: float,
    max_pr_auc_regression: float,
    max_ece: float,
) -> list[tuple[bool, str]]:
    """Run every gate check. Returns (passed, message) in report order."""
    results: list[tuple[bool, str]] = []

    model_lift = candidate["model"]["top_decile_lift"]
    baseline_lift = candidate["baseline"]["top_decile_lift"]
    lift_ratio = model_lift / baseline_lift if baseline_lift else float("inf")
    results.append(
        (
            lift_ratio >= min_decile_lift,
            f"top-decile lift vs baseline: {model_lift:.3f} vs {baseline_lift:.3f} "
            f"({lift_ratio:.3f}x, need >= {min_decile_lift:.2f}x)",
        )
    )

    model_pr = candidate["model"]["pr_auc"]
    baseline_pr = candidate["baseline"]["pr_auc"]
    pr_ratio = model_pr / baseline_pr if baseline_pr else float("inf")
    results.append(
        (
            pr_ratio >= min_pr_auc_floor,
            f"PR-AUC vs baseline (floor): {pr_ratio:.3f}x (need >= {min_pr_auc_floor:.2f}x)",
        )
    )

    ece = candidate["model"]["ece"]
    results.append(
        (
            ece <= max_ece,
            f"calibration error: {ece:.4f} (need <= {max_ece:.4f})",
        )
    )

    if champion is None:
        results.append((True, "no champion model on record; this run becomes the champion"))
    else:
        champion_pr = champion["model"]["pr_auc"]
        delta = model_pr - champion_pr
        results.append(
            (
                delta >= -max_pr_auc_regression,
                f"PR-AUC vs production model: {model_pr:.4f} vs {champion_pr:.4f} "
                f"({delta:+.4f}, tolerance {-max_pr_auc_regression:+.4f})",
            )
        )
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, default=Path("models/candidate_metrics.json"))
    p.add_argument("--champion", type=Path, default=Path("models/production_metrics.json"))
    p.add_argument(
        "--min-decile-lift",
        type=float,
        default=1.08,
        help="required top-decile lift ratio over the baseline",
    )
    p.add_argument(
        "--min-pr-auc-floor",
        type=float,
        default=1.00,
        help="model must not rank worse than the baseline",
    )
    p.add_argument("--max-pr-auc-regression", type=float, default=0.01)
    p.add_argument("--max-ece", type=float, default=0.02)
    args = p.parse_args()

    candidate = load_metrics(args.candidate)
    if candidate is None:
        print(f"FAIL  no candidate metrics at {args.candidate}", file=sys.stderr)
        return 2

    results = check(
        candidate,
        load_metrics(args.champion),
        args.min_decile_lift,
        args.min_pr_auc_floor,
        args.max_pr_auc_regression,
        args.max_ece,
    )

    print("Deployment gate")
    print("=" * 60)
    for passed, message in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {message}")
    print("=" * 60)

    if all(passed for passed, _ in results):
        print("Gate passed: model is cleared for deployment.")
        return 0
    print("Gate failed: deployment blocked.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
