"""Fixtures playing the synthetic warehouse in tests/cassettes/ as query cassettes.

Nothing here is recorded HTTP traffic. Each directory under ``cassettes/`` is a
complete ``mart_gtfs`` for one scenario, a parquet file per table, and the SQL
the loaders build is run against it verbatim rather than replayed from a
transcript -- so a cassette answers the query the loader actually issued, and the
templates stay under test.

The queries are BigQuery dialect, so they are transpiled to DuckDB on the way in.

**A test sees a cassette going in and a metric coming out, and nothing between.**
The fixtures are the metrics themselves -- ``travel_time_variability``,
``expected_wait`` and ``on_time_performance`` -- each one the whole run, from the
first query through to the reported table. ``src.data_loading`` is private as far as these tests are
concerned: no test names a loader, holds one of their frames, or asserts anything
about their shape, so how the data is loaded can change freely as long as the
metrics still come out the same.
"""

from __future__ import annotations

import datetime
import functools
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd
import pytest
import sqlglot
from jinja2 import Environment, FileSystemLoader

from src.data_loading import (
    get_matching_shapes_feeds,
    get_schedule_feed_entries,
    get_schedule_rt_headways,
    get_schedule_rt_stop_times,
)
from src.metric_calculation import (
    OnTimePerformance,
    calculate_expected_wait,
    calculate_on_time_performance,
    calculate_travel_time_variability,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASSETTE_ROOT = Path(__file__).parent / "cassettes"

# one directory per scenario, as written by cassettes/generate.py. a scenario is
# any directory holding parquet, which skips __pycache__ and lets a new one be
# picked up just by generating it
CASSETTES = {
    path.name: path
    for path in sorted(CASSETTE_ROOT.iterdir())
    if path.is_dir() and any(path.glob("*.parquet"))
}

# the schema the templates qualify every table with
WAREHOUSE_SCHEMA = "mart_gtfs"

# what the notebook passes through to the templates. TARGET_DATES is not here: a
# cassette covers whichever dates it covers, so it is read off the cassette
# itself rather than pinned to one date
CONSTANT_MACROS = {
    "TARGET_ROUTE_SHORT_NAME": "R1",
    "TARGET_SCHEDULE_FEED": "Test Schedule",
}

# the cassettes are deliberately small, and dropping a thin shape is not what
# they are here to pin down, so the filter never fires
MIN_TRIPS = 1


class Cassette:
    """One scenario's tables, answering the queries a loader issues against them."""

    def __init__(self, path: Path):
        self.path = path
        self.connection = duckdb.connect()
        # BigQuery hands back timestamps in UTC; DuckDB would otherwise render
        # them in whatever timezone the machine running the tests is set to
        self.connection.execute("set TimeZone = 'UTC'")
        self.connection.execute(f"create schema {WAREHOUSE_SCHEMA}")
        for parquet in sorted(self.path.glob("*.parquet")):
            self.connection.execute(
                f"create view {WAREHOUSE_SCHEMA}.{parquet.stem} as "
                f"select * from read_parquet('{parquet}')"
            )

    def query(self, sql: str) -> pd.DataFrame:
        """Answer one BigQuery statement out of the cassette."""
        duckdb_sql = sqlglot.transpile(sql, read="bigquery", write="duckdb")[0]
        return self.connection.execute(duckdb_sql).df()

    def service_dates(self) -> list[datetime.date]:
        """The service dates this cassette covers, in order.

        Read off fct_daily_schedule_feeds, which has one row per feed per date,
        so a cassette spanning two dates is queried over both without any test
        having to know which one it is.
        """
        feeds = self.query(
            f"select date from {WAREHOUSE_SCHEMA}.fct_daily_schedule_feeds"
        )
        return sorted(pd.to_datetime(feeds["date"]).dt.date.unique())


@dataclass(frozen=True)
class MetricInputs:
    """What the loaders produce for one cassette. Private to this module."""

    stop_times: pd.DataFrame
    headways: pd.DataFrame


@functools.cache
def _cassette(name: str) -> Cassette:
    # building the DuckDB views costs more than the tests do, so do it once
    return Cassette(CASSETTES[name])


@functools.cache
def _metric_inputs(cassette_name: str) -> MetricInputs:
    """Run the loaders end to end against one cassette.

    The loaders only ever hand the engine to ``pd.read_sql_query``, so pointing
    that at the cassette is enough -- and it means the run fails loudly rather
    than reaching for BigQuery if a loader ever grows a real engine call.

    Cached, because the queries cost far more than the metrics do and the same
    cassette is replayed by every test in a class.
    """
    cassette = _cassette(cassette_name)
    env = Environment(loader=FileSystemLoader(REPO_ROOT / "templates"))
    macros = {
        **CONSTANT_MACROS,
        "TARGET_DATES": [date.isoformat() for date in cassette.service_dates()],
    }
    engine = object()

    with mock.patch(
        "pandas.read_sql_query",
        side_effect=lambda sql, con, *args, **kwargs: cassette.query(sql),
    ):
        matching = get_matching_shapes_feeds(engine, env, macros, min_trips=MIN_TRIPS)
        feed_dates = get_schedule_feed_entries(engine, env, matching.feed_dates)
        shape_ids = matching.shapes["shape_id"].to_list()
        loaded = {"feed_dates": feed_dates, "shape_ids": shape_ids}
        return MetricInputs(
            stop_times=get_schedule_rt_stop_times(engine, env, macros, **loaded),
            headways=get_schedule_rt_headways(engine, env, macros, **loaded),
        )


# the loaded frames are cached and shared, so each run gets its own copy -- a
# metric that quietly mutated its input would otherwise leak into the next test
def run_travel_time_variability(cassette_name: str, **kwargs) -> pd.DataFrame:
    """Load a cassette and report trip duration variability over it."""
    return calculate_travel_time_variability(
        _metric_inputs(cassette_name).stop_times.copy(), **kwargs
    )


def run_expected_wait(cassette_name: str, **kwargs) -> pd.DataFrame:
    """Load a cassette and report expected wait over it."""
    return calculate_expected_wait(
        _metric_inputs(cassette_name).headways.copy(), **kwargs
    )


def run_on_time_performance(cassette_name: str, **kwargs) -> OnTimePerformance:
    """Load a cassette and report on time performance over it."""
    return calculate_on_time_performance(
        _metric_inputs(cassette_name).stop_times.copy(), **kwargs
    )


@pytest.fixture
def cassette_name() -> str:
    """Which scenario to play.

    A test pins its own with
    ``@pytest.mark.parametrize("cassette_name", ["variable_headway"])``, or runs
    against all of them with ``CASSETTES``.
    """
    return "constant_headway"


@pytest.fixture
def travel_time_variability(cassette_name: str) -> pd.DataFrame:
    return run_travel_time_variability(cassette_name)


@pytest.fixture
def expected_wait(cassette_name: str) -> pd.DataFrame:
    return run_expected_wait(cassette_name)


@pytest.fixture
def on_time_performance(cassette_name: str) -> OnTimePerformance:
    return run_on_time_performance(cassette_name)
