import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

M_PER_MILE = 1609.344
SECONDS_PER_HOUR = 3600


def plot_distance_over_time(
    trips: list[pd.DataFrame],
    signal_distances: pd.Series,
    nearside_stop_distances: pd.Series,
    farside_stop_distances: pd.Series,
    signal_delays_s: pd.Series | None = None,
    trips_smoothed: list[pd.DataFrame] | None = None,
    highlight_delay_threshold_s: float = 1.0,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one or more trips' distance along the shape vs elapsed time.

    Each trip's time axis is normalized to seconds elapsed from that trip's
    first observation so trips can be overlaid. Smoothed trajectories (if
    provided) are paired with each raw trip by list position. Signals and
    stops are drawn as shape-fixed vertical reference lines; per-signal
    delays are labeled only when signal_delays_s is supplied.

    Args:
        trips: One or more trip dataframes, each with columns
            'distance_along_shape', 'event_time_datetime', and 'TRIP_KEY'.
        signal_distances: Signal distances along the shape (meters).
        nearside_stop_distances: Near-side stop distances along the shape.
        farside_stop_distances: Far-side stop distances along the shape.
        signal_delays_s: Optional per-signal delay (seconds) used for delay
            labels and emphasis. Omit (e.g. in multi-trip mode) to skip
            labels and draw all signal lines uniformly.
        trips_smoothed: Optional smoothed-trajectory dataframes; must align
            with `trips` by position.
        highlight_delay_threshold_s: Signals with delay above this value are
            emphasized. Only applied when signal_delays_s is provided.
        title: Optional axis title. Defaults to a sensible value based on
            the number of trips.

    Returns:
        (figure, axes) tuple for further customization or saving.
    """
    fig, ax = plt.subplots(figsize=(4, 10))
    n_trips = len(trips)
    line_alpha = 0.5 if n_trips > 1 else 1.0

    for i, trip in enumerate(trips):
        trip_start = trip["event_time_datetime"].min()
        elapsed_s = (trip["event_time_datetime"] - trip_start).dt.total_seconds()
        ax.plot(
            trip["distance_along_shape"] / M_PER_MILE,
            elapsed_s,
            marker="o", markersize=3, linestyle="None",
            alpha=line_alpha,
        )
        if trips_smoothed is not None:
            smoothed = trips_smoothed[i]
            elapsed_s_smoothed = (smoothed["event_time_datetime"] - trip_start).dt.total_seconds()
            ax.plot(
                smoothed["distance_along_shape_smoothed"] / M_PER_MILE,
                elapsed_s_smoothed,
                color="red", linewidth=1.5, alpha=line_alpha,
            )

    for i, (signal_id, d) in enumerate(signal_distances.items()):
        delay_s = signal_delays_s.loc[signal_id] if signal_delays_s is not None else None
        is_high_delay = delay_s is not None and delay_s > highlight_delay_threshold_s
        ax.axvline(
            x=d / M_PER_MILE,
            color="black" if is_high_delay else "gray",
            linewidth=1.5 if is_high_delay else 0.5,
            linestyle="-",
            alpha=0.8 if is_high_delay else 0.6,
            label="signal" if i == 0 else None,
        )
        if delay_s is not None:
            ax.text(
                d / M_PER_MILE, 0.98, f"{delay_s:.0f}s",
                transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right",
                fontsize=8 if is_high_delay else 7,
                color="black" if is_high_delay else "gray",
                fontweight="bold" if is_high_delay else "normal",
            )

    for i, d in enumerate(nearside_stop_distances):
        ax.axvline(
            x=d / M_PER_MILE, color="blue", linewidth=0.5, linestyle="--",
            alpha=0.6, label="near-side stop" if i == 0 else None,
        )
    for i, d in enumerate(farside_stop_distances):
        ax.axvline(
            x=d / M_PER_MILE, color="pink", linewidth=0.5, linestyle="--",
            alpha=0.6, label="far-side stop" if i == 0 else None,
        )

    if title is None:
        if n_trips == 1:
            trip_key = trips[0]["TRIP_KEY"].iloc[0]
            title = f"Trip {trip_key} — distance over elapsed time"
        else:
            title = f"{n_trips} trips — distance over elapsed time"

    ax.set_xlabel("Distance along shape (mi)")
    ax.set_ylabel("Elapsed time from trip start (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_speed_over_time(
    trips_speeds: list[pd.DataFrame],
    signal_crossing_times_elapsed_s: pd.Series | None = None,
    signal_delays_s: pd.Series | None = None,
    nearside_stop_crossing_times_elapsed_s: pd.Series | None = None,
    farside_stop_crossing_times_elapsed_s: pd.Series | None = None,
    highlight_delay_threshold_s: float = 1.0,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one or more trips' smoothed speed vs elapsed time.

    Each trip's time axis is normalized to seconds elapsed from that trip's
    first observation so trips can be overlaid. Signal and stop crossing
    times are drawn as vertical reference lines only when provided —
    omit them in multi-trip mode since crossing times vary per trip.

    Args:
        trips_speeds: One or more per-trip speed dataframes, each with
            columns 'speed_m_per_s', 'event_time_datetime', and 'TRIP_KEY'.
        signal_crossing_times_elapsed_s: Optional elapsed seconds when the
            bus crossed each signal, indexed by signal id.
        signal_delays_s: Optional per-signal delay (seconds) used for
            labels and emphasis. Requires signal_crossing_times_elapsed_s
            to be set.
        nearside_stop_crossing_times_elapsed_s: Optional elapsed-seconds
            equivalent for near-side stops.
        farside_stop_crossing_times_elapsed_s: Optional elapsed-seconds
            equivalent for far-side stops.
        highlight_delay_threshold_s: Signals with delay above this value
            are emphasized.
        title: Optional axis title. Defaults to a sensible value based on
            the number of trips.

    Returns:
        (figure, axes) tuple.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    n_trips = len(trips_speeds)
    line_alpha = 0.5 if n_trips > 1 else 1.0

    for trip_speeds in trips_speeds:
        trip_start = trip_speeds["event_time_datetime"].min()
        elapsed_s = (trip_speeds["event_time_datetime"] - trip_start).dt.total_seconds()
        speed_mph = trip_speeds["speed_m_per_s"] * SECONDS_PER_HOUR / M_PER_MILE
        ax.plot(elapsed_s, speed_mph, color="red", linewidth=1.5, alpha=line_alpha)

    if signal_crossing_times_elapsed_s is not None:
        for i, (signal_id, crossing_elapsed) in enumerate(signal_crossing_times_elapsed_s.items()):
            delay_s = signal_delays_s.loc[signal_id] if signal_delays_s is not None else None
            is_high_delay = delay_s is not None and delay_s > highlight_delay_threshold_s
            ax.axvline(
                x=crossing_elapsed,
                color="black" if is_high_delay else "gray",
                linewidth=1.5 if is_high_delay else 0.5,
                linestyle="-",
                alpha=0.8 if is_high_delay else 0.6,
                label="signal" if i == 0 else None,
            )
            if delay_s is not None:
                ax.text(
                    crossing_elapsed, 0.98, f"{delay_s:.0f}s",
                    transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right",
                    fontsize=8 if is_high_delay else 7,
                    color="black" if is_high_delay else "gray",
                    fontweight="bold" if is_high_delay else "normal",
                )

    if nearside_stop_crossing_times_elapsed_s is not None:
        for i, crossing_elapsed in enumerate(nearside_stop_crossing_times_elapsed_s):
            ax.axvline(
                x=crossing_elapsed, color="blue", linewidth=0.5, linestyle="--",
                alpha=0.6, label="near-side stop" if i == 0 else None,
            )
    if farside_stop_crossing_times_elapsed_s is not None:
        for i, crossing_elapsed in enumerate(farside_stop_crossing_times_elapsed_s):
            ax.axvline(
                x=crossing_elapsed, color="pink", linewidth=0.5, linestyle="--",
                alpha=0.6, label="far-side stop" if i == 0 else None,
            )

    if title is None:
        if n_trips == 1:
            trip_key = trips_speeds[0]["TRIP_KEY"].iloc[0]
            title = f"Trip {trip_key} — smoothed speed over elapsed time"
        else:
            title = f"{n_trips} trips — smoothed speed over elapsed time"

    ax.set_xlabel("Elapsed time from trip start (s)")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
