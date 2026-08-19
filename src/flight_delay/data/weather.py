"""Daily weather observations from NOAA, joined onto flights by origin airport.

Weather is the single strongest publicly-available signal for delay risk that is not
already implicit in the schedule, and adding it is the main feature-engineering step
in this project. NOAA's `access/services/data` endpoint is keyless and returns GHCN
daily summaries, which is enough for the fields that matter here: precipitation,
snowfall, temperature range and wind.

Only the busiest airports are mapped. Flights out of unmapped airports keep NaN
weather; the model handles missing values natively rather than imputing, so coverage
can grow later without changing anything downstream.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

NOAA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# Airport IATA code -> NOAA GHCN-Daily station at (or adjacent to) that airport.
AIRPORT_STATIONS: dict[str, str] = {
    "ATL": "USW00013874",
    "LAX": "USW00023174",
    "ORD": "USW00094846",
    "DFW": "USW00003927",
    "DEN": "USW00003017",
    "JFK": "USW00094789",
    "SFO": "USW00023234",
    "SEA": "USW00024233",
    "LAS": "USW00023169",
    "MCO": "USW00012815",
    "EWR": "USW00014734",
    "CLT": "USW00013881",
    "PHX": "USW00023183",
    "MIA": "USW00012839",
    "IAH": "USW00012960",
    "BOS": "USW00014739",
    "MSP": "USW00014922",
    "DTW": "USW00094847",
    "FLL": "USW00012849",
    "PHL": "USW00013739",
    "LGA": "USW00014732",
    "BWI": "USW00093721",
    "SLC": "USW00024127",
    "DCA": "USW00013743",
    "SAN": "USW00023188",
    "MDW": "USW00014819",
    "TPA": "USW00012842",
    "PDX": "USW00024229",
    "STL": "USW00013994",
    "HNL": "USW00022521",
}

WEATHER_FIELDS = ["PRCP", "SNOW", "TMAX", "TMIN", "AWND"]


def fetch_station_month(station: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily summaries for one station. Returns an empty frame on no data."""
    params = {
        "dataset": "daily-summaries",
        "stations": station,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dataTypes": ",".join(WEATHER_FIELDS),
        "format": "json",
        "units": "metric",
    }
    resp = requests.get(NOAA_URL, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json() if resp.text.strip() else []
    if not payload:
        log.warning("no NOAA data for station %s", station)
        return pd.DataFrame(columns=["STATION", "DATE", *WEATHER_FIELDS])
    return pd.DataFrame(payload)


def fetch_weather(start: date, end: date, stations: dict[str, str] | None = None) -> pd.DataFrame:
    """Fetch all mapped stations for a date range, returned keyed by airport code."""
    stations = stations or AIRPORT_STATIONS
    frames = []
    for airport, station in stations.items():
        try:
            raw = fetch_station_month(station, start, end)
        except requests.RequestException as exc:
            # One flaky station must not fail the whole ingest; the join tolerates gaps.
            log.warning("NOAA fetch failed for %s/%s: %s", airport, station, exc)
            continue
        if raw.empty:
            continue
        raw = raw.assign(Origin=airport)
        frames.append(raw)

    if not frames:
        return pd.DataFrame(columns=["Origin", "FlightDate", *WEATHER_FIELDS])

    df = pd.concat(frames, ignore_index=True)
    for field in WEATHER_FIELDS:
        # NOAA pads values with spaces and omits fields a station does not report.
        df[field] = pd.to_numeric(df.get(field), errors="coerce")
    df["FlightDate"] = pd.to_datetime(df["DATE"])
    return df[["Origin", "FlightDate", *WEATHER_FIELDS]]


def attach_weather(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join weather onto flights by (origin airport, date), keeping every flight."""
    if weather.empty:
        for field in WEATHER_FIELDS:
            flights[field] = pd.NA
        return flights
    merged = flights.merge(weather, on=["Origin", "FlightDate"], how="left")
    coverage = merged["PRCP"].notna().mean()
    log.info("weather coverage: %.1f%% of flights", 100 * coverage)
    return merged
