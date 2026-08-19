"""Feature contract for the flight-delay model.

This module is the single source of truth for *what the model is allowed to see*.

The reason it exists as its own module — rather than a list buried in the training
script — is target leakage. The BTS on-time dataset ships the outcome and dozens of
post-hoc columns (actual departure time, taxi time, cause-of-delay breakdowns) in the
same row as the schedule. Any of them would push offline metrics close to perfect and
make the deployed model useless, because at prediction time the flight has not taken
off yet.

`tests/test_schema.py` asserts that the two sets below never intersect and that the
training frame never carries a leaking column, so the guarantee is enforced by CI
instead of by reviewer attention.
"""

from __future__ import annotations

TARGET = "ArrDel15"

# Known before the aircraft pushes back — legitimate model inputs.
#
# Split by cardinality because the two groups need different encodings:
# there are 14 carriers and 19 departure-time blocks, but 330 airports — above the
# 255-category cap on scikit-learn's native categorical support. See features.py.
LOW_CARDINALITY_FEATURES = [
    "Reporting_Airline",
    "DepTimeBlk",
]

HIGH_CARDINALITY_FEATURES = [
    "Origin",
    "Dest",
]

CATEGORICAL_FEATURES = LOW_CARDINALITY_FEATURES + HIGH_CARDINALITY_FEATURES

NUMERIC_FEATURES = [
    "Month",
    "DayOfWeek",
    "DayofMonth",
    "CRSDepHour",
    "CRSArrHour",
    "Distance",
    "CRSElapsedTime",
    # Weather at the origin airport on the day of departure.
    # NOTE: we join *observed* weather. In production this must come from a forecast
    # API, since observations are not available at booking time. See README
    # "Known limitations" — this makes offline metrics optimistic by roughly 1-2 pts
    # of PR-AUC and it would be dishonest to leave it unstated.
    "PRCP",
    "SNOW",
    "TMAX",
    "TMIN",
    "AWND",
    # Derived from the schedule corpus by ScheduleContextTransformer — see that module
    # for why these exist and why they live inside the fitted pipeline.
    "OriginHourlyDepartures",
    "SchedulePaddingRatio",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Only knowable after departure or arrival. Must never reach the model.
LEAKING_COLUMNS = [
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrivalDelayGroups",
    "ActualElapsedTime",
    "AirTime",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    "FirstDepTime",
    "TotalAddGTime",
    "LongestAddGTime",
    "DivArrDelay",
    "DivActualElapsedTime",
]

# Raw BTS columns we read off disk: features plus the fields needed to derive them.
BTS_SOURCE_COLUMNS = [
    "FlightDate",
    "Month",
    "DayOfWeek",
    "DayofMonth",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "DepTimeBlk",
    "Distance",
    "CRSElapsedTime",
    "Cancelled",
    "Diverted",
    TARGET,
]
