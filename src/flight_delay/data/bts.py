"""Download and normalise BTS On-Time Performance data.

One monthly zip is ~26 MB compressed / ~230 MB raw and holds every domestic US
flight for that month (~540k rows for January 2025).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from flight_delay.schema import BTS_SOURCE_COLUMNS, TARGET

log = logging.getLogger(__name__)

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def download_month(year: int, month: int, dest_dir: Path) -> Path:
    """Fetch one month of BTS data. Returns the zip path; skips the download if cached."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"bts_{year}_{month:02d}.zip"
    if target.exists() and target.stat().st_size > 0:
        log.info("cache hit for %s", target.name)
        return target

    url = BASE_URL.format(year=year, month=month)
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        tmp = target.with_suffix(".partial")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        # Rename only after a complete write so an interrupted run cannot leave a
        # truncated file that later looks like a valid cache hit.
        tmp.rename(target)
    return target


def load_month(zip_path: Path) -> pd.DataFrame:
    """Read the CSV inside the zip, keeping only scheduled/outcome columns."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                usecols=BTS_SOURCE_COLUMNS,
                dtype={"Reporting_Airline": "string", "Origin": "string", "Dest": "string"},
                encoding="latin-1",
            )
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the model cannot be scored against and derive scheduling features.

    Cancelled and diverted flights have no arrival delay, so they carry no label. They
    are a genuinely different prediction problem (will this flight be cancelled?) and
    mixing them in would blur the target definition.
    """
    n_before = len(df)
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    df = df.dropna(subset=[TARGET])

    # CRSDepTime is an integer clock reading: 659 -> 06:59, 1735 -> 17:35.
    df = df.assign(
        FlightDate=pd.to_datetime(df["FlightDate"]),
        CRSDepHour=(df["CRSDepTime"] // 100).clip(0, 23).astype("int16"),
        CRSArrHour=(df["CRSArrTime"] // 100).clip(0, 23).astype("int16"),
        ArrDel15=df[TARGET].astype("int8"),
    )
    log.info(
        "cleaned %d -> %d rows (%.1f%% dropped)",
        n_before,
        len(df),
        100 * (1 - len(df) / max(n_before, 1)),
    )
    return df.drop(columns=["Cancelled", "Diverted", "CRSDepTime", "CRSArrTime"])
