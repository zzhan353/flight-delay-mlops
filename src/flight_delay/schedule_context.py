"""Schedule-derived features that the historical-rate baseline cannot express.

The first trained model beat the lookup-table baseline by only 3% on PR-AUC, with
essentially identical ROC-AUC. That is a diagnosis, not a disappointment: the baseline
already conditions on carrier, origin airport and departure hour, and the model's
features barely went beyond those three axes, so there was nothing left to learn.

The two features here describe *mechanisms* the baseline has no way to see:

* **Airport congestion** — how many departures that airport schedules in that hour on
  that weekday. Delay propagates through saturated banks of flights; an airport running
  40 departures at 17:00 behaves differently from one running 4, even for the same
  carrier at the same hour.

* **Schedule padding** — how much block time the airline allots this route compared with
  the median carrier on the same route. Padding is a deliberate airline choice and a
  tightly scheduled flight has less slack to absorb an upstream problem.

Both are implemented as a fitted transformer rather than a precomputed table on disk.
The lookups are learned during `fit` and travel inside the model artifact, so serving a
single flight needs no access to the schedule corpus and cannot drift out of sync with
what training saw — the same training/serving-skew argument that keeps the encoders in
the pipeline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CONGESTION_KEYS = ["Origin", "DayOfWeek", "CRSDepHour"]
ROUTE_KEYS = ["Origin", "Dest"]


class ScheduleContextTransformer(BaseEstimator, TransformerMixin):
    """Adds `OriginHourlyDepartures` and `SchedulePaddingRatio` to the frame."""

    def fit(self, X: pd.DataFrame, y=None) -> ScheduleContextTransformer:
        # Average number of departures per (airport, weekday, hour) slot. Dividing by
        # the number of distinct dates turns a raw count into a per-day rate, so the
        # value does not depend on how many months of history were loaded.
        n_dates = X["FlightDate"].dt.date.nunique() / 7 if "FlightDate" in X else 1
        counts = X.groupby(CONGESTION_KEYS).size().rename("departures")
        self.congestion_ = (counts / max(n_dates, 1)).astype(float)
        self.congestion_median_ = float(self.congestion_.median())

        self.route_block_ = X.groupby(ROUTE_KEYS)["CRSElapsedTime"].median()
        self.global_block_ = float(X["CRSElapsedTime"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        congestion = self.congestion_.reindex(
            pd.MultiIndex.from_frame(out[CONGESTION_KEYS])
        ).to_numpy()
        out["OriginHourlyDepartures"] = pd.Series(congestion, index=out.index).fillna(
            self.congestion_median_
        )

        route_block = self.route_block_.reindex(
            pd.MultiIndex.from_frame(out[ROUTE_KEYS])
        ).to_numpy()
        route_block = pd.Series(route_block, index=out.index).fillna(self.global_block_)
        # >1 means this flight is given more block time than the route's median, i.e.
        # more slack; <1 means a tighter-than-typical schedule.
        out["SchedulePaddingRatio"] = (
            out["CRSElapsedTime"] / route_block.replace(0, pd.NA)
        ).fillna(1.0)
        return out

    def get_feature_names_out(self, input_features=None):
        return list(input_features or []) + ["OriginHourlyDepartures", "SchedulePaddingRatio"]
