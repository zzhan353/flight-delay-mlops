from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.features import build_model, split_by_time, xy
from flight_delay.schedule_context import ScheduleContextTransformer
from flight_delay.schema import FEATURES, LEAKING_COLUMNS, TARGET


def _frame(months=("2025-01", "2025-02", "2025-03"), rows_per_month=5):
    records = []
    for m in months:
        for i in range(rows_per_month):
            records.append(
                {
                    "FlightDate": pd.Timestamp(f"{m}-0{i % 9 + 1}"),
                    **{f: 1 for f in FEATURES},
                    TARGET: i % 2,
                }
            )
    df = pd.DataFrame(records)
    df["Reporting_Airline"] = "AA"
    df["Origin"] = "JFK"
    df["Dest"] = "LAX"
    df["DepTimeBlk"] = "0600-0659"
    df["CRSElapsedTime"] = 120.0
    return df


def test_split_is_chronological_not_random():
    train, holdout = split_by_time(_frame(), holdout_months=1)
    assert train["FlightDate"].max() < holdout["FlightDate"].min()


def test_split_holds_out_the_requested_number_of_months():
    _, holdout = split_by_time(_frame(), holdout_months=2)
    assert holdout["FlightDate"].dt.to_period("M").nunique() == 2


def test_split_refuses_when_there_is_not_enough_history():
    with pytest.raises(ValueError, match="more than"):
        split_by_time(_frame(months=("2025-01",)), holdout_months=1)


def test_xy_returns_pipeline_inputs_without_the_label():
    """xy returns raw pipeline inputs: declared features minus the two the pipeline
    derives itself, plus the columns needed to derive them."""
    X, y = xy(_frame())
    derived = {"OriginHourlyDepartures", "SchedulePaddingRatio"}
    assert set(X.columns) == (set(FEATURES) - derived) | {"FlightDate"}
    assert TARGET not in X.columns


def test_xy_never_returns_a_leaking_column():
    X, _ = xy(_frame())
    assert not set(X.columns) & set(LEAKING_COLUMNS)


def test_pipeline_produces_the_derived_features():
    """The congestion and padding features must actually reach the model."""
    df = _frame()
    X, y = xy(df)
    transformed = ScheduleContextTransformer().fit(X).transform(X)
    assert "OriginHourlyDepartures" in transformed.columns
    assert "SchedulePaddingRatio" in transformed.columns
    assert transformed["SchedulePaddingRatio"].notna().all()


def test_model_handles_unseen_categories_and_missing_values():
    """Serving sees airport codes and weather gaps that training never contained."""
    df = _frame()
    X, y = xy(df)
    model = build_model(max_iter=5).fit(X, y)

    unseen = X.head(1).copy().astype({"PRCP": "float64"})
    unseen.loc[:, "Origin"] = "ZZZ"  # airport not present in training
    unseen.loc[:, "PRCP"] = np.nan  # station with no weather coverage
    prob = model.predict_proba(unseen)[:, 1]
    assert 0.0 <= prob[0] <= 1.0
