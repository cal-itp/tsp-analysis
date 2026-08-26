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

# the day parts a wait time is reported over, as (name, first hour, last hour
# exclusive) on the clock. contiguous and covering 00:00 to 24:00, so every hour
# lands in exactly one of them
TIME_PERIODS = (
    ("Owl", 0, 4),
    ("Early AM", 4, 7),
    ("AM Peak", 7, 10),
    ("Midday", 10, 15),
    ("PM Peak", 15, 20),
    ("Evening", 20, 24),
)

# in clock order, so the reported table sorts by the clock rather than the
# alphabet -- "AM Peak" would otherwise lead and "Owl" would land mid-morning
TIME_PERIOD_ORDER = [name for name, _, _ in TIME_PERIODS]

# flattened for the lookup, since the periods are contiguous and cover 00:00 to
# 24:00 -- every clock hour has exactly one entry here
HOUR_TO_TIME_PERIOD = {
    hour: name for name, start, end in TIME_PERIODS for hour in range(start, end)
}

# one metric value per stop and day part, on this service pattern. stop_id is
# not in the grain: it is fixed by shape_id and stop_sequence, and rides along
WAIT_GRAIN = ("shape_id", "stop_sequence", "time_period")


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


def calculate_expected_wait(
    schedule_rt_headways: pd.DataFrame,
    *,
    wait_grain: Sequence[str] = WAIT_GRAIN,
) -> pd.DataFrame:
    """Expected and excess wait per stop and day part.

    This is the notebook's ``headway_summary`` with two additions: the scheduled
    headway reported alongside the realtime one, and a time period in the grain.

    Each headway is assigned to a day part by the clock hour of
    ``first_actual_departure`` -- the hour the gap in service opens. That is the
    wall clock, not the GTFS hour, so a trip published as 24:15 on one service
    date is counted in Owl on the following morning.

    Args:
        schedule_rt_headways: one row per consecutive pair of trips at a stop,
            as returned by templates/schedule_rt_headways.sql. Needs
            ``shape_id``, ``stop_sequence``, ``stop_id``,
            ``first_actual_departure``, ``rt_headway_seconds`` and
            ``scheduled_headway_seconds``.
        wait_grain: columns identifying one reported value, and the grain of the
            returned frame.

    Returns:
        One row per stop and day part that saw at least one headway. Day parts
        with no service are absent rather than present and empty.

        Carries ``n_headways``, ``mean_rt_headway_seconds``,
        ``median_rt_headway_seconds``, ``max_rt_headway_seconds``,
        ``rt_headway_variance``, ``mean_scheduled_headway_seconds``,
        ``median_scheduled_headway_seconds``,
        ``scheduled_expected_wait_seconds``, ``rt_expected_wait_seconds``,
        ``excess_wait_seconds`` and ``percent_time_excess_wait``.

        ``scheduled_expected_wait_seconds`` is half the median scheduled
        headway. ``rt_expected_wait_seconds`` is the notebook's variance
        adjusted form, ``mean/2 * (1 + var/mean**2)``, so irregular service
        costs the rider more than half the mean gap.
        ``excess_wait_seconds`` is the realtime one less the scheduled one.

        ``percent_time_excess_wait`` is how much of the day part would have left
        a rider waiting longer than the schedule promised, as a proportion of 1.
        A rider turns up at a random moment inside a gap and waits out what is
        left of it, so what they experience is their own remaining wait rather
        than the gap's length: in a gap of H seconds against a scheduled headway
        of S, only those arriving in its first ``H - S`` seconds end up waiting
        longer than S. Each gap therefore contributes ``max(H - S, 0)`` seconds,
        over the total time spent waiting.

        ``time_period`` is an ordered categorical over TIME_PERIOD_ORDER.

        A day part holding a single headway has no sample variance, so its
        realtime expected wait comes back NaN rather than half the one gap. That
        is the notebook's behaviour, kept deliberately: one observation is not
        enough to say anything about how evenly service ran.
    """
    wait_grain = list(wait_grain)

    headways = schedule_rt_headways.copy()
    # the hour the gap in service opens, by the wall clock. every hour has a
    # period, so this only produces nulls if the column itself is null
    headways["time_period"] = pd.Categorical(
        headways["first_actual_departure"].dt.hour.map(HOUR_TO_TIME_PERIOD),
        categories=TIME_PERIOD_ORDER,
        ordered=True,
    )

    # stop_id is fixed by shape_id and stop_sequence, so grouping on it as well
    # splits nothing further -- it just carries the column through without a
    # second pass to merge it back on
    group_keys = list(wait_grain)
    if "stop_id" not in group_keys and "stop_sequence" in group_keys:
        group_keys.insert(group_keys.index("stop_sequence") + 1, "stop_id")

    # how much of each gap leaves a rider worse off than the schedule promised.
    # a rider arriving in the last S seconds of an over-long gap still gets a
    # bus inside the scheduled headway, so only the first H - S seconds count.
    #
    # the clamp is load bearing: without it a short gap would score a negative
    # and cancel a long one out, reporting nothing wrong with service that
    # alternates evenly either side of its schedule. a null scheduled headway
    # leaves this null, so an unscheduled trip counts toward the time waited but
    # never toward the time waited too long
    headways["over_long_headway_seconds"] = (
        headways["rt_headway_seconds"] - headways["scheduled_headway_seconds"]
    ).clip(lower=0)

    # observed, so a day part the stop saw no service in is absent rather than
    # present with a zero count -- grouping on a categorical would otherwise
    # invent a row for every period on every stop
    summary = (
        headways.groupby(group_keys, observed=True)
        .agg(
            n_headways=("rt_headway_seconds", "size"),
            mean_rt_headway_seconds=("rt_headway_seconds", "mean"),
            median_rt_headway_seconds=("rt_headway_seconds", "median"),
            max_rt_headway_seconds=("rt_headway_seconds", "max"),
            rt_headway_variance=("rt_headway_seconds", "var"),
            mean_scheduled_headway_seconds=("scheduled_headway_seconds", "mean"),
            median_scheduled_headway_seconds=("scheduled_headway_seconds", "median"),
            total_headway_seconds=("rt_headway_seconds", "sum"),
            over_long_headway_seconds=("over_long_headway_seconds", "sum"),
        )
        .reset_index()
    )
    # the headway columns arrive as integer seconds, so max comes back integral
    # while the rest are already floats. report them all the same way
    for column in ["median_rt_headway_seconds", "max_rt_headway_seconds"]:
        summary[column] = summary[column].astype(float)

    # the published schedule is even within a day part, so there is no variance
    # term to add: a rider arriving at random waits half a headway
    summary["scheduled_expected_wait_seconds"] = (
        summary["median_scheduled_headway_seconds"] / 2
    )
    # what actually ran is not even, and the variance term is the penalty for
    # that -- a rider is more likely to turn up during a long gap than a short
    # one, so the average wait exceeds half the average headway
    summary["rt_expected_wait_seconds"] = (
        summary["mean_rt_headway_seconds"]
        / 2
        * (
            1
            + summary["rt_headway_variance"]
            / summary["mean_rt_headway_seconds"] ** 2
        )
    )
    summary["excess_wait_seconds"] = (
        summary["rt_expected_wait_seconds"] - summary["scheduled_expected_wait_seconds"]
    )

    # weighted by seconds rather than counted by gap, because a rider turning up
    # at random is likelier to land in a long gap than a short one
    summary["percent_time_excess_wait"] = (
        summary["over_long_headway_seconds"] / summary["total_headway_seconds"]
    )
    summary = summary.drop(
        columns=["total_headway_seconds", "over_long_headway_seconds"]
    )

    # groupby already sorts by the keys, but say so rather than depend on it --
    # time_period is ordered, so this is clock order and not alphabetical
    return summary.sort_values(group_keys).reset_index(drop=True)
