"""Metric calculations for GTFS-realtime derived TSP metrics."""

from __future__ import annotations

import datetime
from collections.abc import Sequence

import pandas as pd

# a trip_id is only unique within a service date, so a trip is identified by the
# date as well. shape_id is redundant for well formed data, but including it
# stops a trip that somehow appears under two shapes from bridging a travel time
# across them.
TRIP_GRAIN = ("service_date", "shape_id", "feed_key", "trip_id")

# one metric value per segment: the travel time into this stop_sequence, on this
# service pattern
SEGMENT_GRAIN = ("shape_id", "feed_key", "stop_sequence")


def calculate_travel_time_variability(
    schedule_rt_stop_times: pd.DataFrame,
    *,
    upper_percentile: float = 0.8,
    lower_percentile: float = 0.2,
    trip_grain: Sequence[str] = TRIP_GRAIN,
    segment_grain: Sequence[str] = SEGMENT_GRAIN,
) -> pd.DataFrame:
    """Trip duration variability: the p80/p20 ratio of segment travel times.

    A segment is the run from one timepoint to the next, so its travel time is
    the actual arrival at a stop minus the actual departure from the stop before
    it on the same trip. Trips from every selected service date are pooled into a
    single sample per segment.

    Args:
        schedule_rt_stop_times: one row per trip and timepoint, as returned by
            templates/schedule_rt_stop_times.sql. Needs the columns in
            ``trip_grain`` and ``segment_grain`` plus ``actual_arrival_pacific``
            and ``actual_departure_pacific``.
        upper_percentile: numerator of the ratio.
        lower_percentile: denominator of the ratio.
        trip_grain: columns identifying one vehicle trip.
        segment_grain: columns identifying one segment, and the grain of the
            returned frame.

    Returns:
        One row per segment, with columns ``travel_time_variability``,
        ``p80_travel_time_seconds``, ``p20_travel_time_seconds``,
        ``n_rt_trips`` and ``n_service_dates``.

        The first timepoint on a pattern has no preceding stop, so its travel
        time is null and its ratio is NaN.
    """
    trip_grain = list(trip_grain)
    segment_grain = list(segment_grain)

    # groupby().shift() reads rows in their existing order, so sort rather than
    # rely on the query's ORDER BY surviving whatever happened to the frame
    stop_times = schedule_rt_stop_times.sort_values(
        trip_grain + ["stop_sequence"]
    ).copy()

    stop_times["prior_actual_departure_pacific"] = stop_times.groupby(trip_grain)[
        "actual_departure_pacific"
    ].shift()
    stop_times["time_difference"] = (
        stop_times["actual_arrival_pacific"]
        - stop_times["prior_actual_departure_pacific"]
    )
    # a negative travel time means the feed disagreed with itself about the order
    # of two stops, so drop it rather than let it into the percentiles
    stop_times["time_difference_cleaned"] = stop_times["time_difference"].where(
        stop_times["time_difference"] >= datetime.timedelta(0),
        pd.NaT,
    )
    # NOTE: .dt.seconds is the seconds component, not the total, so a segment
    # taking 24 hours or more would wrap. No real timepoint segment comes close.
    stop_times["time_difference_seconds"] = stop_times[
        "time_difference_cleaned"
    ].dt.seconds

    by_segment = stop_times.groupby(segment_grain)
    travel_time = by_segment["time_difference_seconds"]

    p_upper = travel_time.quantile(upper_percentile)
    p_upper.name = f"p{upper_percentile * 100:.0f}_travel_time_seconds"
    p_lower = travel_time.quantile(lower_percentile)
    p_lower.name = f"p{lower_percentile * 100:.0f}_travel_time_seconds"

    variability = p_upper / p_lower
    variability.name = "travel_time_variability"

    n_trips_per_timepoint = by_segment["trip_id"].count()
    n_trips_per_timepoint.name = "n_rt_trips"

    # a shape only covers the dates its feed was active on, so record how many of
    # the selected dates actually fed each segment
    n_dates_per_timepoint = by_segment["service_date"].nunique()
    n_dates_per_timepoint.name = "n_service_dates"

    return pd.concat(
        [
            variability,
            p_upper,
            p_lower,
            n_trips_per_timepoint,
            n_dates_per_timepoint,
        ],
        axis=1,
    ).reset_index()
