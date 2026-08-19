"""Build the modelling dataset: download BTS months, join weather, write parquet.

    python -m flight_delay.data.build --months 2025-01 2025-02

Each month is written as its own parquet file. Keeping months separate (rather than
one growing file) is what makes the monthly drift check cheap: the monitor reads the
newest month and compares it against the training window without rescanning history.
"""

from __future__ import annotations

import argparse
import calendar
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from flight_delay.data.bts import download_month, load_month
from flight_delay.data.weather import attach_weather, fetch_weather
from flight_delay.schema import FEATURES, LEAKING_COLUMNS, TARGET

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_month(value: str) -> tuple[int, int]:
    """Parse a YYYY-MM string into (year, month)."""
    try:
        year_s, month_s = value.split("-")
        year, month = int(year_s), int(month_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {value!r}") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"month out of range in {value!r}")
    return year, month


def build_month(year: int, month: int, raw_dir: Path, with_weather: bool = True) -> pd.DataFrame:
    """Produce one month of model-ready rows."""
    zip_path = download_month(year, month, raw_dir)
    flights = load_month(zip_path)

    if with_weather:
        last_day = calendar.monthrange(year, month)[1]
        weather = fetch_weather(date(year, month, 1), date(year, month, last_day))
        flights = attach_weather(flights, weather)
    else:
        for column in ("PRCP", "SNOW", "TMAX", "TMIN", "AWND"):
            flights[column] = pd.NA

    return assert_no_leakage(flights)


def assert_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if a post-departure column survived into the modelling frame.

    This runs on every build, not just in tests, because the failure it guards against
    is silent: a leaked column produces excellent offline metrics and a worthless model.
    """
    leaked = sorted(set(df.columns) & set(LEAKING_COLUMNS))
    if leaked:
        raise ValueError(
            f"post-departure columns reached the modelling frame: {leaked}. "
            "These are unknown at prediction time; see flight_delay.schema."
        )
    # Two features are produced inside the model pipeline, not by the data build.
    from flight_delay.schedule_context import ScheduleContextTransformer

    derived = set(ScheduleContextTransformer().get_feature_names_out([]))
    missing = sorted(set(FEATURES) - set(df.columns) - derived)
    if missing:
        raise ValueError(f"declared features missing from frame: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True, metavar="YYYY-MM")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--no-weather", action="store_true", help="skip the NOAA join")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for value in args.months:
        year, month = parse_month(value)
        df = build_month(year, month, args.raw_dir, with_weather=not args.no_weather)
        out = args.out_dir / f"flights_{year}_{month:02d}.parquet"
        df.to_parquet(out, index=False)
        rate = df[TARGET].mean()
        log.info("wrote %s  rows=%d  delay_rate=%.3f", out, len(df), rate)


if __name__ == "__main__":
    main()
