import numpy as np
import pandas as pd

STOPPED_SPEED_THRESHOLD_MPS = 2 * 1609.344 / 3600  # 2 mph in m/s #TODO: maybe this should be based on percentiles around the stop area nistead
MIN_STOPPED_RUN_SECONDS = 5


def find_stopped_events(
    smoothed_trajectory: pd.Series,
    smoothed_speeds_mps: pd.Series,
) -> pd.DataFrame:
    """Run-length-encode slow samples into discrete stopped events.

    A stopped event is a maximal run of consecutive samples below
    STOPPED_SPEED_THRESHOLD_MPS lasting longer than MIN_STOPPED_RUN_SECONDS.

    Args:
        smoothed_trajectory: Bus distance (meters) along the matched shape,
            indexed by time (DatetimeIndex or numeric seconds).
        smoothed_speeds_mps: Bus speed (m/s), aligned positionally to
            `smoothed_trajectory`.

    Returns:
        DataFrame with one row per stopped event and columns duration_s,
        start_distance_m, end_distance_m. Empty if there are no qualifying events.
    """
    trajectory_times = smoothed_trajectory.index
    if isinstance(trajectory_times, pd.DatetimeIndex):
        time_seconds_array = (
            trajectory_times - trajectory_times[0]
        ).total_seconds().to_numpy()
    else:
        time_seconds_array = np.asarray(trajectory_times, dtype=float)

    samples_df = pd.DataFrame({
        "time_s": time_seconds_array,
        "distance_m": smoothed_trajectory.to_numpy(dtype=float),
        "is_slow": (smoothed_speeds_mps < STOPPED_SPEED_THRESHOLD_MPS).to_numpy(),
    })
    samples_df["run_id"] = (
        samples_df["is_slow"] != samples_df["is_slow"].shift()
    ).cumsum()

    slow_runs_grouped = samples_df[samples_df["is_slow"]].groupby("run_id")
    stopped_events = pd.DataFrame({
        "start_time_s": slow_runs_grouped["time_s"].first(),
        "end_time_s": slow_runs_grouped["time_s"].last(),
        "duration_s": slow_runs_grouped["time_s"].last() - slow_runs_grouped["time_s"].first(),
        "start_distance_m": slow_runs_grouped["distance_m"].first(),
        "end_distance_m": slow_runs_grouped["distance_m"].last(),
    })
    return stopped_events[
        stopped_events["duration_s"] > MIN_STOPPED_RUN_SECONDS
    ].reset_index(drop=True)


def identify_nearside_signals(
    signals_projected: pd.Series,
    stops_projected: pd.Series,
    stops_nearside: pd.Series,
    nearside_stop_distance: float,
) -> pd.Series:
    """Flag signals within `nearside_stop_distance` of any near-side stop.

    Such signals' stopped time is dominated by passenger dwell at the near-side
    stop rather than signal delay, so callers typically exclude them.

    Args:
        signals_projected: Distance (meters) along the shape of each signal,
            indexed by signal identifier.
        stops_projected: Distance (meters) along the shape of each stop.
        stops_nearside: Near-side flag per stop, indexed like stops_projected.
            Values not exactly True are treated as not near-side.
        nearside_stop_distance: Max signal-to-near-side-stop distance (meters)
            for the signal to be flagged.

    Returns:
        Boolean Series indexed like `signals_projected`.
    """
    signal_distances_array = signals_projected.to_numpy()
    nearside_stop_distances_array = stops_projected[stops_nearside == True].to_numpy()

    if nearside_stop_distances_array.size == 0:
        return pd.Series(False, index=signals_projected.index)

    # Note - we have manually tagged stops as near-side to handle edge cases, so
    # we use abs to exclude even if the stop is slightly behind the stop line.
    min_signal_to_nearside_stop_distance = np.abs(
        signal_distances_array[:, None] - nearside_stop_distances_array[None, :]
    ).min(axis=1)
    return pd.Series(
        min_signal_to_nearside_stop_distance <= nearside_stop_distance,
        index=signals_projected.index,
    )


def estimate_intersection_time_single_trip(
        smoothed_trajectory: pd.Series,
        smoothed_speeds_mps: pd.Series,
        stops_projected: pd.Series,
        stops_nearside: pd.Series,
        signals_projected: pd.Series,
        nearside_stop_distance: int,
        signal_effect_distance: int,
    ) -> pd.Series:
    """
    Estimate intersection time for a single trip. Uses the following methodology:

    Methodology:
        1. Determine any signals that are within nearside_stop_distance of a marked nearside stop, and exclude them
        2. Determine the range of impact that each signal has, by finding the intervals before signal_effect_distance
        3. Find all cases where a bus has become stopped using the smothed trajectory, and count the time that the bus has been stopped for. Associate all such times with each signal
        a. For now, just define this as being time when a bus is below 2mph for more than 5 seconds
        4. Return a series with the same index as signals_projected, showing the amount of time the bus is stopped at each signal. Return 0 for nearside signals.

    Parameters:
        smoothed_trajectory: Bus distance (meters) along the matched shape,
            indexed by time. The index may be a DatetimeIndex or a numeric
            (seconds) index. Recommended sampling frequency is at least one
            sample per 0.5 seconds. Distance should be monotonically
            non-decreasing along the trip.
        smoothed_speeds_mps: Bus speed in meters per second, indexed
            identically to `smoothed_trajectory` (one speed value per
            trajectory sample).
        stops_projected: Distance (meters) along the same matched shape of
            each GTFS stop served by the trip's route. Indexed by stop
            identifier.
        stops_nearside: Flag indicating whether each stop is a near-side
            stop, indexed identically to `stops_projected`. Values that
            are not exactly `True` (including False and NaN) are treated
            as not near-side.
        signals_projected: Distance (meters) along the same matched shape
            of each traffic signal of interest. Points should represent stop bars.
            Indexed by signal identifier. The returned Series uses this index.
        nearside_stop_distance: Maximum distance (meters) between a signal
            and any near-side stop for the signal to be excluded from the
            output (forced to 0). Intended to filter out signals whose
            stopped time is dominated by passenger dwell at a near-side
            stop rather than by signal delay.
        signal_effect_distance: Length (meters) of the upstream zone in
            which a stopped event is attributed to a signal. For a signal
            at distance d, stopped events whose end-distance falls in
            [d - signal_effect_distance, d] count toward that signal.

    Returns:
        pd.Series indexed identically to `signals_projected`, with values
        equal to the total time (seconds) the bus was stopped within each
        signal's upstream effect zone. Signals near a near-side stop
        are returned as 0.
    """
    signal_distances_array = signals_projected.to_numpy()

    # Step 1: identify signals within nearside_stop_distance of any nearside-marked stop
    is_signal_excluded = identify_nearside_signals(
        signals_projected, stops_projected, stops_nearside, nearside_stop_distance
    )

    # Steps 2-3: run-length-encode slow samples to find stopped events
    stopped_events = find_stopped_events(smoothed_trajectory, smoothed_speeds_mps)

    # Step 4: assignment of stopped-event durations to each signal's upstream range
    if stopped_events.empty or signal_distances_array.size == 0:
        signal_stopped_time_s = pd.Series(0.0, index=signals_projected.index)
    else:
        event_distances_array = stopped_events["end_distance_m"].to_numpy()
        event_durations_array = stopped_events["duration_s"].to_numpy()
        # event_in_signal_range[i, j] = True if event j ends in signal i's upstream range
        event_in_signal_range = (
            (event_distances_array[None, :] >= signal_distances_array[:, None] - signal_effect_distance)
            & (event_distances_array[None, :] <= signal_distances_array[:, None])
        )
        # assign stopped events to signals using a dot product
        per_signal_stopped_time_s = event_in_signal_range.astype(float).dot(event_durations_array)
        signal_stopped_time_s = pd.Series(per_signal_stopped_time_s, index=signals_projected.index)

    # for now, we don't attribute nearside stop delay to signals, so the result is 0
    signal_stopped_time_s[is_signal_excluded] = 0.0

    return signal_stopped_time_s