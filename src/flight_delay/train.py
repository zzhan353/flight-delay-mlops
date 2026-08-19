"""Train the delay classifier and record everything needed to reproduce the run.

    python -m flight_delay.train --backend local
    python -m flight_delay.train --backend azureml     # submit to an AML compute cluster

Why two backends: at this data scale (~500k rows/month, ~90 s to fit) a GitHub Actions
runner trains the model for free, while an Azure ML cluster costs $0.29/hour plus a
10-minute environment image build on first use. So CI trains locally and Azure ML is
used for the things it is uniquely good at — model registry, lineage and experiment
comparison. The `azureml` backend exists so the same code scales up unchanged when the
data no longer fits that argument.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from flight_delay.baseline import HistoricalRateBaseline
from flight_delay.features import build_model, calibrate, split_by_time, split_calibration, xy
from flight_delay.metrics import evaluate
from flight_delay.schema import FEATURES

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models/candidate")


def load_dataset(processed_dir: Path) -> pd.DataFrame:
    """Concatenate every processed month, oldest first."""
    files = sorted(processed_dir.glob("flights_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no parquet files in {processed_dir}; run `python -m flight_delay.data.build` first"
        )
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    log.info("loaded %d rows from %d month(s)", len(df), len(files))
    return df.sort_values("FlightDate").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["local", "azureml"], default="local")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--metrics-out", type=Path, default=Path("models/candidate_metrics.json"))
    p.add_argument("--holdout-months", type=int, default=1)
    p.add_argument(
        "--calibration-days",
        type=int,
        default=14,
        help="trailing days of the training window reserved for calibration",
    )
    p.add_argument(
        "--no-calibration",
        action="store_true",
        help="skip post-hoc calibration (used to demonstrate the gate catching it)",
    )
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--l2-regularization", type=float, default=0.0)
    p.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="train on a fraction of rows; used by the fast PR check",
    )
    p.add_argument("--experiment", default="flight-delay")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def configure_tracking(backend: str, experiment: str) -> None:
    """Point MLflow at the right tracking store for the chosen backend."""
    if backend == "azureml":
        import os

        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        client = MLClient(
            DefaultAzureCredential(),
            os.environ["AZURE_SUBSCRIPTION_ID"],
            os.environ["AZURE_RESOURCE_GROUP"],
            os.environ["AZURE_ML_WORKSPACE"],
        )
        mlflow.set_tracking_uri(client.workspaces.get(client.workspace_name).mlflow_tracking_uri)
    mlflow.set_experiment(experiment)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_tracking(args.backend, args.experiment)

    df = load_dataset(args.processed_dir)
    train_df, holdout_df = split_by_time(df, args.holdout_months)
    if args.sample_frac < 1.0:
        train_df = train_df.sample(frac=args.sample_frac, random_state=42)
    train_span = (train_df["FlightDate"].min().date(), train_df["FlightDate"].max().date())
    holdout_span = (holdout_df["FlightDate"].min().date(), holdout_df["FlightDate"].max().date())
    log.info(
        "train %s..%s (%d rows) | holdout %s..%s (%d rows)",
        *train_span,
        len(train_df),
        *holdout_span,
        len(holdout_df),
    )

    X_train, y_train = xy(train_df)
    X_holdout, y_holdout = xy(holdout_df)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "learning_rate": args.learning_rate,
                "max_iter": args.max_iter,
                "max_leaf_nodes": args.max_leaf_nodes,
                "min_samples_leaf": args.min_samples_leaf,
                "l2_regularization": args.l2_regularization,
                "holdout_months": args.holdout_months,
                "sample_frac": args.sample_frac,
                "n_features": len(FEATURES),
                "train_rows": len(train_df),
                "train_end": str(train_df["FlightDate"].max().date()),
                "holdout_start": str(holdout_df["FlightDate"].min().date()),
            }
        )

        # The lookup-table baseline is fitted on the same data and scored on the same
        # holdout, so the comparison is apples to apples.
        baseline = HistoricalRateBaseline().fit(train_df)
        baseline_metrics = evaluate(y_holdout, baseline.predict_proba(holdout_df))

        model = build_model(
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
        )

        if args.no_calibration:
            model.fit(X_train, y_train)
            uncalibrated_metrics = None
        else:
            fit_df, calib_df = split_calibration(train_df, args.calibration_days)
            X_fit, y_fit = xy(fit_df)
            X_calib, y_calib = xy(calib_df)
            log.info(
                "fit on %d rows, calibrate on %d rows (last %d days)",
                len(fit_df),
                len(calib_df),
                args.calibration_days,
            )
            model.fit(X_fit, y_fit)
            # Record what the model looked like before calibration so the run history
            # shows the effect rather than just the final number.
            uncalibrated_metrics = evaluate(y_holdout, model.predict_proba(X_holdout)[:, 1])
            mlflow.log_metric("uncalibrated_ece", uncalibrated_metrics["ece"])
            mlflow.log_metric("uncalibrated_pr_auc", uncalibrated_metrics["pr_auc"])
            model = calibrate(model, X_calib, y_calib)

        model_metrics = evaluate(y_holdout, model.predict_proba(X_holdout)[:, 1])

        for name, value in model_metrics.items():
            mlflow.log_metric(name, value)
        for name, value in baseline_metrics.items():
            mlflow.log_metric(f"baseline_{name}", value)

        lift = model_metrics["pr_auc"] / baseline_metrics["pr_auc"]
        mlflow.log_metric("pr_auc_vs_baseline", lift)

        signature = infer_signature(X_holdout.head(100), model.predict_proba(X_holdout.head(100)))
        # Serialization format: cloudpickle, deliberately.
        #
        # MLflow defaults to skops, which cannot execute arbitrary code on load and is
        # the safer choice — but skops cannot serialize custom classes, and
        # ScheduleContextTransformer is exactly that. The alternative would be to move
        # congestion/padding lookups outside the model into a table the serving code
        # joins itself, which reintroduces the training/serving skew the pipeline
        # design exists to prevent.
        #
        # The residual risk is accepted with a stated boundary: model artifacts are
        # produced only by this repository's CI, stored in a registry the service
        # authenticates to, and never loaded from user-supplied paths. If the artifact
        # store were ever shared or externally writable, this decision would have to be
        # revisited — deserialising cloudpickle from an untrusted source is remote code
        # execution.
        signature = infer_signature(X_holdout.head(100), model.predict_proba(X_holdout.head(100)))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_holdout.head(2),
            serialization_format="cloudpickle",
        )

        args.model_dir.parent.mkdir(parents=True, exist_ok=True)
        if args.model_dir.exists():
            import shutil

            shutil.rmtree(args.model_dir)
        mlflow.sklearn.save_model(model, str(args.model_dir), serialization_format="cloudpickle")

        report = {
            "run_id": run.info.run_id,
            "calibrated": not args.no_calibration,
            "uncalibrated": uncalibrated_metrics,
            "model": model_metrics,
            "baseline": baseline_metrics,
            "pr_auc_vs_baseline": lift,
            "train_rows": len(train_df),
            "holdout_start": str(holdout_df["FlightDate"].min().date()),
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(report, indent=2))

    print(f"\n{'metric':<18}{'baseline':>12}{'model':>12}{'change':>12}")
    print("-" * 54)
    for key in ("pr_auc", "roc_auc", "brier", "ece", "top_decile_lift"):
        b, m = baseline_metrics[key], model_metrics[key]
        print(f"{key:<18}{b:>12.4f}{m:>12.4f}{m - b:>+12.4f}")
    print(f"\nPR-AUC is {lift:.2f}x the baseline; wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
