"""A deliberately dumb baseline the ML model has to beat.

"The model gets 0.42 PR-AUC" means nothing on its own. The question a reviewer should
ask is: better than *what*? Any airline ops team already knows that evening departures
out of Newark run late. If a gradient-boosted model cannot beat a lookup table of
historical delay rates, it is not worth the deployment, the monitoring or the on-call
burden — and shipping it anyway is how ML teams lose credibility.

This baseline is therefore a first-class part of the pipeline, logged alongside the
model on every run, and the deployment gate is expressed as a margin over it.
"""

from __future__ import annotations

import pandas as pd

from flight_delay.schema import TARGET

GROUP_KEYS = ["Reporting_Airline", "Origin", "CRSDepHour"]


class HistoricalRateBaseline:
    """Predicts the historical delay rate for (carrier, origin airport, departure hour).

    Falls back to coarser groupings, and finally the global rate, for combinations not
    seen during training — the same unseen-category problem the encoder handles.
    """

    def __init__(self, min_samples: int = 30) -> None:
        self.min_samples = min_samples
        self.global_rate_: float | None = None

    def fit(self, df: pd.DataFrame) -> HistoricalRateBaseline:
        self.global_rate_ = float(df[TARGET].mean())
        grouped = df.groupby(GROUP_KEYS)[TARGET].agg(["mean", "size"])
        # Groups with too few flights are noise, not signal; drop them so the fallback
        # supplies a more stable estimate instead.
        self.rates_ = grouped.loc[grouped["size"] >= self.min_samples, "mean"]
        carrier = df.groupby("Reporting_Airline")[TARGET].mean()
        self.carrier_rates_ = carrier
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self.global_rate_ is None:
            raise RuntimeError("baseline must be fitted before predicting")
        index = pd.MultiIndex.from_frame(df[GROUP_KEYS])
        probs = pd.Series(self.rates_.reindex(index).to_numpy(), index=df.index)
        fallback = df["Reporting_Airline"].map(self.carrier_rates_)
        return probs.fillna(fallback).fillna(self.global_rate_)
