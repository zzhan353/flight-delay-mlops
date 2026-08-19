"""Model construction.

Three decisions here carry most of the engineering weight:

1. **Preprocessing lives inside the estimator.** The encoders are fitted as part of the
   same `Pipeline` object that gets logged to the model registry, so the serving code
   receives raw fields (carrier code, airport code, scheduled hour) and never has to
   reproduce a transformation. Training/serving skew is not a bug you find in tests —
   it produces plausible-looking wrong answers in production — so the only reliable
   fix is to make it structurally impossible.

2. **Two encodings, chosen by cardinality.** Carrier (14) and departure block (19) go
   through an ordinal encoder and are declared native categoricals. Airports (330 of
   them) exceed scikit-learn's 255-category limit, so they are target-encoded instead.
   Target encoding is the right tool rather than merely a workaround: an airport's
   historical delay propensity *is* the signal we want from it, and one-hot encoding
   330 columns would be both larger and weaker.

   The leakage risk in target encoding is real — replacing a category with a statistic
   derived from the label famously overfits. `TargetEncoder` avoids it with internal
   cross-fitting: each training row is encoded using folds that exclude it, while the
   fitted transformer uses the full-data mapping at predict time.

3. **HistGradientBoostingClassifier over LightGBM/XGBoost.** It handles NaNs natively
   (34% of rows have no weather join), it is already a scikit-learn dependency, and it
   keeps the serving image small — which is what makes the sub-$1/month hosting budget
   achievable. On this data it performs within noise of LightGBM, so the extra binary
   dependency buys nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from flight_delay.schedule_context import ScheduleContextTransformer
from flight_delay.schema import (
    FEATURES,
    HIGH_CARDINALITY_FEATURES,
    LOW_CARDINALITY_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


def build_model(
    learning_rate: float = 0.1,
    max_iter: int = 300,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 50,
    l2_regularization: float = 0.0,
    random_state: int = 42,
) -> Pipeline:
    """Assemble the preprocessing + estimator pipeline."""
    ordinal = OrdinalEncoder(
        # Carrier codes change between months (airlines merge, new entrants appear);
        # an unseen code at serving time must not raise.
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
        dtype=np.float64,
    )
    target_encoder = TargetEncoder(
        target_type="binary",
        smooth="auto",  # shrinks small airports toward the global rate
        cv=5,  # cross-fitting: this is what keeps the encoding leak-free
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("low_cardinality", ordinal, LOW_CARDINALITY_FEATURES),
            ("high_cardinality", target_encoder, HIGH_CARDINALITY_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        # Only the ordinal-encoded block is categorical; target-encoded airports are
        # continuous. Positions follow the ColumnTransformer output order.
        categorical_features=list(range(len(LOW_CARDINALITY_FEATURES))),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return Pipeline(
        [
            ("schedule_context", ScheduleContextTransformer()),
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_by_time(df: pd.DataFrame, holdout_months: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: train on earlier months, evaluate on the latest ones.

    A random split would let the model learn from flights that happen *after* the ones
    it is scored on. Delay behaviour is strongly seasonal and propagates within a day,
    so a random split inflates offline metrics and tells you nothing about how the
    model will do next month — which is the only question that matters in production.
    """
    if "FlightDate" not in df.columns:
        raise ValueError("FlightDate is required for a chronological split")
    periods = np.sort(df["FlightDate"].dt.to_period("M").unique())
    if len(periods) <= holdout_months:
        raise ValueError(
            f"need more than {holdout_months} month(s) of data to hold out; got {len(periods)}"
        )
    cutoff = periods[-holdout_months]
    is_holdout = df["FlightDate"].dt.to_period("M") >= cutoff
    return df.loc[~is_holdout].copy(), df.loc[is_holdout].copy()


# Columns the pipeline needs as raw input even though they are not model features:
# ScheduleContextTransformer derives its lookups from them.
PIPELINE_INPUT_EXTRAS = ["FlightDate"]


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the pipeline's input columns and the label.

    Declared features minus the two the pipeline derives itself, plus the raw columns
    ScheduleContextTransformer needs to compute them.
    """
    derived = set(ScheduleContextTransformer().get_feature_names_out([]))
    columns = [c for c in FEATURES if c not in derived] + PIPELINE_INPUT_EXTRAS
    return df[columns], df[TARGET].astype(int)


def split_calibration(
    train_df: pd.DataFrame, calibration_days: int = 14
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve the most recent slice off the training window to calibrate on.

    Boosted trees rank well but are badly calibrated out of the box: they push
    probabilities away from the base rate, so a "74% chance of delay" prediction
    corresponds to flights that are actually late 38% of the time. The demo shows the
    probability to a user, so ranking alone is not enough.

    The calibration slice must be held out from fitting (calibrating on data the model
    trained on just re-learns the overconfidence) and must not come from the evaluation
    holdout (that would leak the test set). Taking the tail of the training window keeps
    the whole pipeline chronological and mirrors how a retrain would work in production:
    fit on history, calibrate on the most recent weeks, deploy forward.
    """
    cutoff = train_df["FlightDate"].max() - pd.Timedelta(days=calibration_days)
    fit_part = train_df[train_df["FlightDate"] <= cutoff]
    calib_part = train_df[train_df["FlightDate"] > cutoff]
    if len(calib_part) == 0 or len(fit_part) == 0:
        raise ValueError(
            f"calibration_days={calibration_days} leaves an empty split; "
            f"training window is {train_df['FlightDate'].min().date()}"
            f"..{train_df['FlightDate'].max().date()}"
        )
    return fit_part.copy(), calib_part.copy()


def calibrate(model: Pipeline, X_calib: pd.DataFrame, y_calib: pd.Series) -> CalibratedClassifierCV:
    """Wrap a fitted pipeline in an isotonic calibrator.

    Isotonic rather than Platt scaling: the miscalibration here is not a monotone
    sigmoid distortion (it under-predicts at the low end *and* over-predicts at the
    high end), and with ~200k calibration rows there is more than enough data for the
    non-parametric fit that would otherwise overfit.
    """
    # FrozenEstimator replaces the removed cv="prefit": it tells the calibrator that
    # `model` is already fitted and must not be refitted on the calibration slice.
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    return calibrated.fit(X_calib, y_calib)
