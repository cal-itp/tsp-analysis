"""Metric calculations for GTFS-realtime derived TSP metrics."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd

# a trip_id is only unique within a service date, so a trip is identified by the
# date as well. shape_id is redundant for well formed data, but including it
# stops a trip that somehow appears under two shapes from bridging a travel time
# across them.
TRIP_GRAIN = ("service_date", "shape_id", "feed_key", "trip_id")

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

# one metric value per segment and day part: the travel time into this
# stop_sequence, on this service pattern, in this part of the day
SEGMENT_GRAIN = ("shape_id", "feed_key", "stop_sequence", "time_period")

# one metric value per stop and day part, on this service pattern. stop_id is
# not in the grain: it is fixed by shape_id and stop_sequence, and rides along
WAIT_GRAIN = ("shape_id", "stop_sequence", "time_period")

# on time performance is reported at two grains at once: where along the line
# punctuality goes wrong, and how the service pattern did overall in a day part
PUNCTUALITY_STOP_GRAIN = ("shape_id", "stop_sequence", "time_period")
PUNCTUALITY_PERIOD_GRAIN = ("shape_id", "time_period")

# what counts as early and what counts as late, in seconds off the published
# arrival. both are inclusive, so a vehicle exactly this early or exactly this
# late is not on time -- the transit industry's usual "one minute early, five late"
EARLY_SECONDS = 60
LATE_SECONDS = 5 * 60


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
    single sample per segment and day part.

    Each run is assigned to a day part by the clock hour of the actual arrival
    that ends it. Realtime throughout, unlike on time performance's scheduled
    hour: this metric reads no schedule at all, so a trip the schedule does not
    cover is placed on the clock and measured like any other. A run that starts
    in one day part and ends in the next is reported in the one it ended in.

    Args:
        schedule_rt_stop_times: one row per trip and timepoint, as returned by
            templates/schedule_rt_stop_times.sql. Needs the columns in
            ``trip_grain`` and ``segment_grain`` bar ``time_period``, plus
            ``actual_arrival_pacific`` and ``actual_departure_pacific``.
        upper_percentile: numerator of the ratio.
        lower_percentile: denominator of the ratio.
        trip_grain: columns identifying one vehicle trip.
        segment_grain: columns identifying one segment, and the grain of the
            returned frame.

    Returns:
        One row per segment and day part that saw service, with columns
        ``travel_time_variability``, ``p80_travel_time_seconds``,
        ``p20_travel_time_seconds``, ``n_rt_trips`` and ``n_service_dates``. Day
        parts with no service are absent rather than present and empty.

        ``time_period`` is an ordered categorical over TIME_PERIOD_ORDER.

        The first timepoint on a pattern has no preceding stop, so its travel
        time is null and its ratio is NaN. It is still reported, and still
        counts its trips, because the day part comes from its own arrival.

        ``n_service_dates`` counts the dates that fed the segment *in that day
        part*, so a window whose dates each contribute one day part reports 1
        everywhere -- the day part is a clock hour, while the service date is
        what a trip is listed under.

        A stop time with no realtime arrival has no day part to fall in and is
        dropped. It carries no travel time either way, so this only moves
        ``n_rt_trips``.
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

    # the hour the vehicle actually reached the stop, by the wall clock. every
    # hour has a period, so this only produces nulls if the arrival itself is
    # null -- and those rows are dropped by the grouping below
    stop_times["time_period"] = pd.Categorical(
        stop_times["actual_arrival_pacific"].dt.hour.map(HOUR_TO_TIME_PERIOD),
        categories=TIME_PERIOD_ORDER,
        ordered=True,
    )

    # observed, so a day part the segment saw no service in is absent rather than
    # present with a zero count -- grouping on a categorical would otherwise
    # invent a row for every period on every segment
    by_segment = stop_times.groupby(segment_grain, observed=True)
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

    summary = pd.concat(
        [
            variability,
            p_upper,
            p_lower,
            n_trips_per_timepoint,
            n_dates_per_timepoint,
        ],
        axis=1,
    ).reset_index()

    # time_period is ordered, so this runs down the line and then round the clock
    # rather than the alphabet
    return summary.sort_values(segment_grain).reset_index(drop=True)


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
        * (1 + summary["rt_headway_variance"] / summary["mean_rt_headway_seconds"] ** 2)
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


class OnTimePerformance(NamedTuple):
    """On time performance reported at both of its grains."""

    by_stop: pd.DataFrame
    """One row per stop and day part: where along the line punctuality goes."""

    by_period: pd.DataFrame
    """One row per day part, pooling every timepoint on the pattern."""


def _summarise_punctuality(
    arrivals: pd.DataFrame,
    group_keys: list[str],
    *,
    median_percentile: float,
    upper_percentile: float,
) -> pd.DataFrame:
    """One punctuality row per group, at whatever grain ``group_keys`` names."""
    # observed, so a day part a stop saw no arrival in is absent rather than
    # present with a zero count -- grouping on a categorical would otherwise
    # invent a row for every period on every stop
    grouped = arrivals.groupby(group_keys, observed=True)

    summary = grouped.agg(
        n_timepoint_arrivals=("absolute_deviation_seconds", "size"),
    )

    # the percentiles are of the *absolute* deviation, so a stop where half the
    # service runs early and half runs late reports how far off it was, not the
    # zero the two sides would average out to
    deviation = grouped["absolute_deviation_seconds"]
    for percentile in [median_percentile, upper_percentile]:
        summary[f"p{percentile * 100:.0f}_absolute_deviation_seconds"] = (
            deviation.quantile(percentile)
        )

    # the mean of a boolean is the proportion of it that is true. every
    # proportion is over timepoint arrivals rather than trips: a trip late at
    # every stop counts once per stop, which is what makes the pooled table more
    # than a restatement of the per-stop one
    for column in ["is_early", "is_late", "is_not_on_time"]:
        summary[column.replace("is_", "proportion_")] = grouped[column].mean()

    # the headline number, reported alongside its complement so a table can lead
    # with the share of service that made its published time rather than the
    # share that missed it
    summary["schedule_adherence"] = 1 - summary["proportion_not_on_time"]

    return summary.reset_index().sort_values(group_keys).reset_index(drop=True)


def calculate_on_time_performance(
    schedule_rt_stop_times: pd.DataFrame,
    *,
    early_seconds: int = EARLY_SECONDS,
    late_seconds: int = LATE_SECONDS,
    median_percentile: float = 0.5,
    upper_percentile: float = 0.8,
    stop_grain: Sequence[str] = PUNCTUALITY_STOP_GRAIN,
    period_grain: Sequence[str] = PUNCTUALITY_PERIOD_GRAIN,
) -> OnTimePerformance:
    """On time performance at every timepoint: how far off the schedule service ran.

    The deviation is the arrival the vehicle actually made less the arrival the
    schedule published for that trip at that stop, signed so that late is
    positive. A vehicle is **early** if it beats its published arrival by
    ``early_seconds`` or more, **late** if it misses it by ``late_seconds`` or
    more, and **not on time** if it is either. Both comparisons are inclusive,
    so a vehicle exactly a minute early is not on time.

    Arrivals rather than departures: this asks when the vehicle reached the
    stop, so a bus that pulled in on time and then stood too long is on time
    here. The wait metrics measure departure to departure instead.

    Every value is reported twice, at two grains. The per-stop table says where
    along the line punctuality goes -- a trip that leaves on time and loses six
    minutes by the end of the run shows up at the last timepoints only. The
    pooled one says how the pattern did overall in a day part. The pooled
    percentiles are taken over the arrivals, not over the per-stop numbers, so
    they are not the average of the table above them.

    Args:
        schedule_rt_stop_times: one row per trip and timepoint, as returned by
            templates/schedule_rt_stop_times.sql. Needs the columns in
            ``stop_grain`` and ``period_grain`` bar ``time_period``, plus
            ``stop_id``, ``arrival_deviation_seconds`` and
            ``scheduled_arrival_pacific``.
        early_seconds: beat the published arrival by this or more to be early.
        late_seconds: miss it by this or more to be late.
        median_percentile: the lower reported percentile of absolute deviation.
        upper_percentile: the upper one.
        stop_grain: columns identifying one reported per-stop value.
        period_grain: columns identifying one reported pooled value.

    Returns:
        An :class:`OnTimePerformance` pair. Both frames carry
        ``n_timepoint_arrivals``, ``p50_absolute_deviation_seconds``,
        ``p80_absolute_deviation_seconds``, ``proportion_early``,
        ``proportion_late``, ``proportion_not_on_time`` and
        ``schedule_adherence``, which is one less that last one -- the share of
        arrivals that made their published time; ``by_stop`` also carries
        ``stop_sequence`` and ``stop_id``. The percentile column names follow the
        percentiles asked for.

        Every proportion is of the timepoint arrivals in the group, not of the
        trips. A trip running late at three timepoints is three late arrivals,
        so the pooled proportions weight it three times as heavily as a trip
        that went early at one stop only.

        ``time_period`` is an ordered categorical over TIME_PERIOD_ORDER, taken
        from the hour of the **scheduled** arrival -- a trip running late is
        reported in the day part it was promised in, and moving the boundary
        cannot move a trip between day parts.

        Timepoints the published schedule has no arrival for are dropped: there
        is nothing to measure them against. They are still counted by the travel
        time metric, which needs no schedule.
    """
    stop_grain = list(stop_grain)
    period_grain = list(period_grain)

    # stop_id is fixed by shape_id and stop_sequence, so grouping on it as well
    # splits nothing further -- it just carries the column through without a
    # second pass to merge it back on
    stop_keys = list(stop_grain)
    if "stop_id" not in stop_keys and "stop_sequence" in stop_keys:
        stop_keys.insert(stop_keys.index("stop_sequence") + 1, "stop_id")

    # a realtime arrival with no published counterpart cannot be measured
    # against one. this is the same left join the travel time metric relies on
    # to keep an unscheduled trip's run time, so the drop belongs here rather
    # than in the query
    arrivals = schedule_rt_stop_times.loc[
        schedule_rt_stop_times["arrival_deviation_seconds"].notna()
    ].copy()

    # the hour the vehicle was *due*, by the wall clock. the query has already
    # put the published seconds back on the clock, so a trip published as 24:15
    # on one service date is counted in Owl on the following morning
    arrivals["time_period"] = pd.Categorical(
        arrivals["scheduled_arrival_pacific"].dt.hour.map(HOUR_TO_TIME_PERIOD),
        categories=TIME_PERIOD_ORDER,
        ordered=True,
    )

    arrivals["absolute_deviation_seconds"] = arrivals["arrival_deviation_seconds"].abs()
    # inclusive either side, so the thresholds themselves are not on time
    arrivals["is_early"] = arrivals["arrival_deviation_seconds"] <= -early_seconds
    arrivals["is_late"] = arrivals["arrival_deviation_seconds"] >= late_seconds
    # a vehicle cannot be both, so this is the two proportions added rather than
    # anything that could overlap
    arrivals["is_not_on_time"] = arrivals["is_early"] | arrivals["is_late"]

    percentiles = {
        "median_percentile": median_percentile,
        "upper_percentile": upper_percentile,
    }
    return OnTimePerformance(
        by_stop=_summarise_punctuality(arrivals, stop_keys, **percentiles),
        by_period=_summarise_punctuality(arrivals, period_grain, **percentiles),
    )
