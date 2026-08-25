"""Warehouse queries backing the GTFS-realtime derived TSP metrics."""

from __future__ import annotations

from typing import NamedTuple

import geopandas as gpd
import pandas as pd
import shapely.geometry
import shapely.wkt
from jinja2 import Environment
from sqlalchemy.engine import Engine


class MatchingShapes(NamedTuple):
    """The service patterns a metric run covers, and the feeds behind them."""

    shapes: gpd.GeoDataFrame
    """One row per shape and feed, with the shape geometry and a trip count."""

    feed_dates: pd.DataFrame
    """Which feed was in effect on each service date, one row per pair."""

    feed_keys: list[str]
    """The distinct feeds, for filtering later queries."""


def make_linestring(x: list[str]) -> shapely.geometry.LineString:
    # shapely errors if the array contains only one point
    if len(x) > 1:
        # each point in the array is wkt
        # so convert them to shapely points via list comprehension
        as_wkt = [shapely.wkt.loads(i) for i in x]
        return shapely.geometry.LineString([point.coords[0] for point in as_wkt])
    return shapely.geometry.LineString()


def get_matching_shapes_feeds(
    engine: Engine,
    env: Environment,
    constant_macros: dict,
    *,
    min_trips: int,
) -> MatchingShapes:
    """Find the service patterns running the target route over the target dates.

    Runs templates/shapes_routes.sql to find which shapes the route ran under,
    then templates/shape_geometry.sql for their geometry. Shapes are collapsed to
    one row per shape and feed, with trips summed over every selected date, and
    those carrying fewer than ``min_trips`` are dropped as too thin to report on.

    Args:
        engine: SQLAlchemy engine pointed at the warehouse.
        env: Jinja environment loading from templates/.
        constant_macros: TARGET_DATES, TARGET_ROUTE_SHORT_NAME and
            TARGET_SCHEDULE_FEED, as rendered into shapes_routes.sql.
        min_trips: minimum trips over the whole window for a shape to be kept.

    Returns:
        A MatchingShapes triple. ``feed_dates`` and ``feed_keys`` are taken
        before the ``min_trips`` filter, so they still describe every feed that
        ran the route in the window.
    """
    # Get the target route and feeds, one row per service date and shape
    matching_shapes_df = pd.read_sql_query(
        env.get_template("shapes_routes.sql").render(constant_macros),
        engine,
    )
    # convert dates from bigquery from datetime to date
    matching_shapes_df["service_date"] = matching_shapes_df["service_date"].dt.date
    # drop dates where the feed ran but the target route did not
    matching_shapes_df = matching_shapes_df.loc[matching_shapes_df["shape_id"].notna()]

    # the feed in effect can change part way through the window, so keep the
    # mapping from each service date to the feed that was active on it
    feed_dates = matching_shapes_df[["service_date", "feed_key"]].drop_duplicates()
    feed_keys = feed_dates["feed_key"].drop_duplicates().to_list()

    # collapse to one row per shape, with trips summed over all the selected dates
    # natural key for shapes is feed key + shape id
    matching_shapes_df["ct"] = matching_shapes_df.groupby("shape_id")["ct"].transform(
        "sum"
    )
    matching_shapes_df = (
        matching_shapes_df.sort_values("service_date", ascending=True)
        .drop_duplicates(subset=["shape_id", "feed_key"], keep="last")
        .rename(columns={"service_date": "last_seen_date"})
        .sort_values("ct", ascending=False)
        .reset_index(drop=True)
    )
    matching_shapes_df = matching_shapes_df.loc[matching_shapes_df["ct"] >= min_trips]

    # Get the geometry separately, reading as few date partitions as possible - a
    # shape does not change within a feed version, so take the last date each
    # shape was active
    geometry_dates = matching_shapes_df["last_seen_date"].unique()
    shape_geometry = pd.read_sql_query(
        env.get_template("shape_geometry.sql").render(
            {
                "GEOMETRY_DATES": geometry_dates,
                "FEED_KEYS": feed_keys,
                "SHAPE_IDS": matching_shapes_df["shape_id"].to_list(),
            }
        ),
        engine,
    )
    matching_shapes_df = matching_shapes_df.merge(
        shape_geometry.drop_duplicates(subset=["shape_id", "feed_key"])[
            ["shape_id", "feed_key", "pt_array"]
        ],
        on=["shape_id", "feed_key"],
        how="left",
        validate="one_to_one",
        # TODO: figure out why there are sometimes na geometries
    ).dropna(subset=["pt_array"])

    matching_shapes_geom = gpd.GeoSeries(
        matching_shapes_df["pt_array"].map(make_linestring), crs=4326
    )
    matching_shapes = gpd.GeoDataFrame(
        matching_shapes_df.drop(["pt_array"], axis=1),
        geometry=matching_shapes_geom,
    )

    return MatchingShapes(
        shapes=matching_shapes,
        feed_dates=feed_dates,
        feed_keys=feed_keys,
    )


def get_schedule_feed_entries(
    engine: Engine,
    env: Environment,
    feed_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Attach each feed's URL and validity start to the dates it was active on.

    Runs templates/schedule_feed_entry.sql for the feeds in ``feed_dates``.
    ``base64_url`` and ``_valid_from`` are what partition the stop times tables,
    so the next query needs them to prune its scan.

    Args:
        engine: SQLAlchemy engine pointed at the warehouse.
        env: Jinja environment loading from templates/.
        feed_dates: the ``feed_dates`` member of a MatchingShapes, one row per
            service date and the feed active on it.

    Returns:
        ``feed_dates`` with ``base64_url`` and ``_valid_from`` added.

    Raises:
        AssertionError: if the warehouse has no entry for some feed, which would
            silently drop that feed's dates from the merge.
    """
    feed_keys = feed_dates["feed_key"].drop_duplicates().to_list()
    schedule_feed_entries = pd.read_sql_query(
        env.get_template("schedule_feed_entry.sql").render({"FEED_KEYS": feed_keys}),
        engine,
    )
    assert schedule_feed_entries.index.size == len(feed_keys)

    # attach each service date to the feed that was active on it, so the stop
    # times query can pair them up rather than mixing feed versions together
    return feed_dates.merge(
        schedule_feed_entries, on="feed_key", validate="many_to_one"
    )


def get_schedule_rt_stop_times(
    engine: Engine,
    env: Environment,
    constant_macros: dict,
    *,
    feed_dates: pd.DataFrame,
    shape_ids: list[str],
) -> pd.DataFrame:
    """Get the actual arrivals and departures at every timepoint on the route.

    Runs templates/schedule_rt_stop_times.sql, which pairs each service date
    with the schedule feed that was active on it, keeps the stops flagged as
    timepoints on every trip, and joins them to the realtime stop times.

    Args:
        engine: SQLAlchemy engine pointed at the warehouse.
        env: Jinja environment loading from templates/.
        constant_macros: TARGET_DATES, TARGET_ROUTE_SHORT_NAME and
            TARGET_SCHEDULE_FEED.
        feed_dates: as returned by :func:`get_schedule_feed_entries`, so
            carrying ``base64_url`` and ``_valid_from``.
        shape_ids: the service patterns to report on.

    Returns:
        One row per trip and timepoint, ready for
        :func:`src.metric_calculation.calculate_travel_time_variability`.
    """
    return pd.read_sql_query(
        env.get_template("schedule_rt_stop_times.sql").render(
            {
                "FEED_DATES": feed_dates.to_dict("records"),
                "FEED_KEYS": feed_dates["feed_key"].drop_duplicates().to_list(),
                "FEED_VALID_FROMS": feed_dates["_valid_from"]
                .drop_duplicates()
                .to_list(),
                "SCHEDULE_BASE64_URLS": feed_dates["base64_url"]
                .drop_duplicates()
                .to_list(),
                "SHAPE_IDS": shape_ids,
                **constant_macros,
            }
        ),
        engine,
    )


def get_schedule_rt_headways(
    engine: Engine,
    env: Environment,
    constant_macros: dict,
    *,
    feed_dates: pd.DataFrame,
    shape_ids: list[str],
) -> pd.DataFrame:
    """Get the scheduled and realtime headway between consecutive trips.

    Runs templates/schedule_rt_headways.sql. A headway is the gap in service at
    a stop: the time from one vehicle pulling out to the next one pulling in.
    The realtime side measures it between the actual departure and the actual
    arrival; the scheduled side is the same gap between the times
    dim_stop_arrivals published for those same two trips. Having both on one row
    is what makes an excess wait computable.

    Unlike :func:`get_schedule_rt_stop_times` this covers every stop on the
    pattern, not only the timepoints.

    Trips are paired within a service pattern and a service date, in the order
    they actually reached the stop. The last trip to call at a stop on a date
    starts no headway, so n trips on a date give n - 1 rows per stop.

    Pairs never span two service dates. GTFS keeps a trip that runs past
    midnight on the date it was listed under, publishing it in hours past 24, so
    dim_stop_arrivals counts each trip's seconds from midnight on its own date
    and the count restarts at the boundary; a pair spanning it would report a
    scheduled headway nearly a day negative. Reporting is by service date rather
    than by calendar date for that reason, and because how an agency splits its
    service between two dates is its own business. The overnight gap between one
    date's last trip and the next date's first is therefore not reported as a
    headway at all. Within a date the pairing is unaffected: a trip departing
    23:45 still pairs with the 24:00 one behind it.

    Args:
        engine: SQLAlchemy engine pointed at the warehouse.
        env: Jinja environment loading from templates/.
        constant_macros: TARGET_DATES, TARGET_ROUTE_SHORT_NAME and
            TARGET_SCHEDULE_FEED.
        feed_dates: as returned by :func:`get_schedule_feed_entries`, so
            carrying ``base64_url`` and ``_valid_from``.
        shape_ids: the service patterns to report on.

    Returns:
        One row per consecutive pair of trips at a stop, keyed by
        ``first_trip_key``, ``second_trip_key`` and ``stop_sequence``, and
        carrying ``service_date``, ``stop_id``, ``shape_id``,
        ``first_actual_departure``, ``second_actual_arrival``,
        ``rt_headway_seconds``, ``first_scheduled_departure_sec``,
        ``second_scheduled_arrival_sec`` and ``scheduled_headway_seconds``.

        ``service_date`` is the date both trips in the pair were listed under,
        which is not necessarily the calendar date either of them ran on.

        The scheduled columns are seconds since midnight on that service date,
        as dim_stop_arrivals stores them, and can exceed 86400 for a trip
        running past midnight. They are null where the schedule does not cover a
        trip the feed reported running; the realtime side is always populated.
    """
    return pd.read_sql_query(
        env.get_template("schedule_rt_headways.sql").render(
            {
                "FEED_DATES": feed_dates.to_dict("records"),
                "FEED_KEYS": feed_dates["feed_key"].drop_duplicates().to_list(),
                "FEED_VALID_FROMS": feed_dates["_valid_from"]
                .drop_duplicates()
                .to_list(),
                "SCHEDULE_BASE64_URLS": feed_dates["base64_url"]
                .drop_duplicates()
                .to_list(),
                "SHAPE_IDS": shape_ids,
                **constant_macros,
            }
        ),
        engine,
    )
