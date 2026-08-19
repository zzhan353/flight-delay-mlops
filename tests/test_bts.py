from __future__ import annotations

import pandas as pd

from flight_delay.data.bts import clean
from flight_delay.schema import TARGET


def _raw_row(**overrides):
    row = {
        "FlightDate": "2025-01-01",
        "Month": 1,
        "DayOfWeek": 3,
        "DayofMonth": 1,
        "Reporting_Airline": "AA",
        "Origin": "JFK",
        "Dest": "LAX",
        "CRSDepTime": 659,
        "CRSArrTime": 1020,
        "DepTimeBlk": "0600-0659",
        "Distance": 2475.0,
        "CRSElapsedTime": 381.0,
        "Cancelled": 0.0,
        "Diverted": 0.0,
        TARGET: 0.0,
    }
    row.update(overrides)
    return row


def test_clean_derives_scheduled_hours():
    df = clean(pd.DataFrame([_raw_row()]))
    assert df.loc[0, "CRSDepHour"] == 6  # 659 -> 06:59
    assert df.loc[0, "CRSArrHour"] == 10


def test_clean_drops_cancelled_and_diverted():
    df = clean(pd.DataFrame([_raw_row(), _raw_row(Cancelled=1.0), _raw_row(Diverted=1.0)]))
    assert len(df) == 1


def test_clean_drops_unlabelled_rows():
    df = clean(pd.DataFrame([_raw_row(), _raw_row(**{TARGET: None})]))
    assert len(df) == 1


def test_clean_removes_raw_clock_columns():
    df = clean(pd.DataFrame([_raw_row()]))
    assert "CRSDepTime" not in df.columns
    assert "Cancelled" not in df.columns


def test_midnight_hour_does_not_overflow():
    df = clean(pd.DataFrame([_raw_row(CRSDepTime=2400, CRSArrTime=5)]))
    assert df.loc[0, "CRSDepHour"] == 23
    assert df.loc[0, "CRSArrHour"] == 0
