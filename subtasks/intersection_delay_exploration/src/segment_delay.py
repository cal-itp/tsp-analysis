"""Segment-level delay breakdown for one shape, per MIT thesis §3.3.

For each signal-bounded segment and each trip, the observed travel time is
decomposed (Eq 3.6):

    T_obs = T_ff + T_dwell + D_signal + D_congestion + Loss

  * T_ff: free-flow travel time, the FREE_FLOW_PERCENTILE-th percentile of night
    segment travel times (Eq 3.10). Night ~ free flow.
  * T_dwell: dwell. We do NOT model door events; stopping activities near a bus
    stop or near a near-side signal are treated as dwell, excluded from the
    signal/congestion attribution, and subtracted here.
  * D_signal = uniform + overflow, measured directly from stop durations (§3.3.5):
      - The stop closest to the downstream signal (within SIGNAL_STOP_AREA_M of
        the stop bar) is uniform delay, capped at the red-phase length
        (RED_PHASE_PERCENTILE-th percentile of night stop-bar stop durations,
        §3.3.6). Any excess over the cap becomes overflow.
      - Additional upstream stops within OVERFLOW_ZONE_M (= 2 * SIGNAL_STOP_AREA_M)
        that are connected to the signal stop by creeping (stop-slow-stop with no
        intervening "faster period" above FASTER_PERIOD_SPEED_MPS) are overflow.
        A slowdown separated by a faster period, or beyond the overflow zone,
        falls into congestion via the residual.
  * D_congestion: residual (Eq 3.21), max(0, T_obs - T_ff - T_dwell - D_signal).
  * Loss is neglected (thesis §3.3.8).

Segments that END at a near-side signal are ignored. All calculation lives here;
the notebook only loads inputs, calls run_segment_delay_analysis, and displays.
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from delay_calculation import find_stopped_events, identify_nearside_signals
from match_shapes_vp import (
    distances_to_crossing_times,
    project_points_on_shape,
    project_vp_on_shape,
)
from smooth_trajectory import compute_speeds_per_trip, smooth_distances_per_trip
from constants import (
    MAX_SHAPE_JUMP_M,
    MAX_SNAP_DISTANCE_M,
    SHAPE_KEY_TO_SHAPE_ID_MAP,
)

NIGHT_START_HOUR = 21          # trips starting at/after 21:00 ...
NIGHT_END_HOUR = 6             # ... or before 06:00 are nighttime (free flow)
FREE_FLOW_PERCENTILE = 5       # percentile of night travel times -> T_ff (Eq 3.10)
RED_PHASE_PERCENTILE = 95      # percentile of night stop-bar stops -> uniform cap (§3.3.6)
SIGNAL_STOP_AREA_M = 100       # uniform-delay zone upstream of the stop bar
OVERFLOW_ZONE_M = 2 * SIGNAL_STOP_AREA_M  # overflow search window upstream of the signal
NEARSIDE_STOP_DISTANCE_M = 30  # signal within this of a near-side stop is itself near-side
DWELL_EXCLUSION_DISTANCE_M = 30  # stop within this of a bus stop / near-side signal = dwell
FASTER_PERIOD_SPEED_MPS = 15 * 1609.344 / 3600  # 15 mph; above this between stops = not overflow
SMOOTH_FREQ_SECONDS = 1.0      # smoothing/speed sample spacing
MIN_BASELINE_TRIPS = 3         # min night samples to trust a per-segment baseline


def classify_trip_period(start_timestamp: pd.Timestamp) -> str:
    """Return 'night' if the trip starts in [21:00, 06:00), else 'day'."""
    hour = start_timestamp.hour
    return "night" if (hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR) else "day"


def build_segments(
    signals_projected: pd.Series,
    signal_is_nearside: pd.Series,
) -> pd.DataFrame:
    """Segments between consecutive signals, dropping those ending at a near-side
    signal.

    Returns a DataFrame indexed by segment_id with start_signal, end_signal,
    start_distance_m, end_distance_m, length_m, ordered along the shape.
    """
    ordered = signals_projected.sort_values()
    signal_ids = ordered.index.to_numpy()
    signal_distances = ordered.to_numpy()

    segment_rows = []
    for i in range(len(ordered) - 1):
        end_signal = signal_ids[i + 1]
        if bool(signal_is_nearside.loc[end_signal]):
            continue  # ignore sections that end at a near-side signal
        segment_rows.append({
            "segment_id": f"sig{signal_ids[i]}_to_sig{end_signal}",
            "start_signal": signal_ids[i],
            "end_signal": end_signal,
            "start_distance_m": signal_distances[i],
            "end_distance_m": signal_distances[i + 1],
            "length_m": signal_distances[i + 1] - signal_distances[i],
        })

    return pd.DataFrame(segment_rows).set_index("segment_id")


def _speeds_by_second(trajectory: pd.Series, speeds: pd.Series) -> pd.Series:
    """Speed (m/s) indexed by seconds-into-trip, matching find_stopped_events' clock."""
    seconds = (trajectory.index - trajectory.index[0]).total_seconds().to_numpy()
    return pd.Series(speeds.to_numpy(dtype=float), index=seconds)


def _max_speed_between(speeds_by_second: pd.Series, start_s: float, end_s: float) -> float:
    """Max speed strictly between two times (seconds). 0 if the window is empty."""
    if end_s <= start_s:
        return 0.0
    window = speeds_by_second[
        (speeds_by_second.index > start_s) & (speeds_by_second.index < end_s)
    ]
    return float(window.max()) if len(window) else 0.0


DELAY_TYPE_COLORS = {
    "dwell": "tab:gray",
    "uniform": "tab:blue",
    "overflow": "tab:purple",
    "congestion": "tab:orange",
}


def classify_segment_stops(
    segment_stops: pd.DataFrame,
    is_dwell: np.ndarray,
    speeds_by_second: pd.Series,
    downstream_distance_m: float,
) -> pd.Series:
    """Label each in-segment stop dwell / uniform / overflow / congestion.

    - dwell: flagged by `is_dwell` (near a bus stop or near-side signal).
    - uniform: the non-dwell stop closest to the downstream signal, within
      SIGNAL_STOP_AREA_M of the stop bar.
    - overflow: non-dwell stops upstream of the uniform stop, within
      OVERFLOW_ZONE_M, reached without a faster period (creeping stop-slow-stop).
    - congestion: every other non-dwell stop.

    Returns a Series of labels indexed like `segment_stops`.
    """
    delay_type = pd.Series("congestion", index=segment_stops.index, dtype=object)
    if len(segment_stops) == 0:
        return delay_type
    delay_type[segment_stops.index[is_dwell]] = "dwell"

    non_dwell_stops = segment_stops[~is_dwell]
    candidates = non_dwell_stops[
        (non_dwell_stops["end_distance_m"] >= downstream_distance_m - OVERFLOW_ZONE_M)
        & (non_dwell_stops["end_distance_m"] <= downstream_distance_m)
    ].sort_values("end_distance_m", ascending=False)  # closest to the signal first

    if candidates.empty:
        return delay_type
    signal_stop = candidates.iloc[0]
    if signal_stop["end_distance_m"] < downstream_distance_m - SIGNAL_STOP_AREA_M:
        # nearest stop is past the signal stop area -> no stop-bar (uniform) stop
        return delay_type
    delay_type[candidates.index[0]] = "uniform"

    previous_stop = signal_stop
    for stop_index in candidates.index[1:]:
        upstream_stop = candidates.loc[stop_index]
        gap_max_speed = _max_speed_between(
            speeds_by_second, upstream_stop["end_time_s"], previous_stop["start_time_s"]
        )
        if gap_max_speed >= FASTER_PERIOD_SPEED_MPS:
            break  # faster period -> remaining upstream slowdowns are congestion
        delay_type[stop_index] = "overflow"
        previous_stop = upstream_stop

    return delay_type


def _signal_delay_components(
    non_dwell_stops: pd.DataFrame,
    speeds_by_second: pd.Series,
    downstream_distance_m: float,
) -> tuple[float, float, bool]:
    """Uniform and overflow stop durations for one trip-segment (uncapped uniform).

    Returns (uniform_stop_duration_s, overflow_stop_duration_s, has_signal_stop).
    """
    delay_type = classify_segment_stops(
        non_dwell_stops,
        np.zeros(len(non_dwell_stops), dtype=bool),
        speeds_by_second,
        downstream_distance_m,
    )
    uniform_stop_duration_s = float(
        non_dwell_stops.loc[delay_type == "uniform", "duration_s"].sum()
    )
    overflow_stop_duration_s = float(
        non_dwell_stops.loc[delay_type == "overflow", "duration_s"].sum()
    )
    return uniform_stop_duration_s, overflow_stop_duration_s, bool((delay_type == "uniform").any())


def _segment_rows_for_trip(
    service_date: str,
    trip_key,
    trajectory: pd.Series,
    speeds: pd.Series,
    signals_projected: pd.Series,
    segments: pd.DataFrame,
    dwell_exclusion_distances: np.ndarray,
) -> list[dict]:
    """Raw per-segment quantities for a single trip (pre-baseline)."""
    period = classify_trip_period(trajectory.index[0])
    stopped_events = find_stopped_events(trajectory, speeds)
    speeds_by_second = _speeds_by_second(trajectory, speeds)
    signal_crossing_times = distances_to_crossing_times(signals_projected, trajectory)

    rows = []
    for segment in segments.itertuples():
        if (
            segment.start_signal not in signal_crossing_times.index
            or segment.end_signal not in signal_crossing_times.index
        ):
            continue

        observed_travel_time_s = (
            signal_crossing_times.loc[segment.end_signal]
            - signal_crossing_times.loc[segment.start_signal]
        ).total_seconds()
        if not np.isfinite(observed_travel_time_s) or observed_travel_time_s <= 0:
            continue

        # Stops in the segment's impact area (downstream of A, up to B].
        in_segment = stopped_events[
            (stopped_events["end_distance_m"] > segment.start_distance_m)
            & (stopped_events["end_distance_m"] <= segment.end_distance_m)
        ]

        if in_segment.empty or dwell_exclusion_distances.size == 0:
            is_dwell = np.zeros(len(in_segment), dtype=bool)
        else:
            distance_to_exclusion = np.abs(
                in_segment["end_distance_m"].to_numpy()[:, None]
                - dwell_exclusion_distances[None, :]
            ).min(axis=1)
            is_dwell = distance_to_exclusion <= DWELL_EXCLUSION_DISTANCE_M

        dwell_s = float(in_segment["duration_s"].to_numpy()[is_dwell].sum())
        non_dwell_stops = in_segment[~is_dwell]

        uniform_stop_duration_s, overflow_stop_duration_s, has_signal_stop = (
            _signal_delay_components(
                non_dwell_stops, speeds_by_second, segment.end_distance_m
            )
        )

        rows.append({
            "service_date": service_date,
            "TRIP_KEY": trip_key,
            "period": period,
            "segment_id": segment.Index,
            "observed_travel_time_s": observed_travel_time_s,
            "dwell_s": dwell_s,
            "uniform_stop_duration_s": uniform_stop_duration_s,
            "overflow_stop_duration_s": overflow_stop_duration_s,
            "has_signal_stop": has_signal_stop,
        })
    return rows


def _segment_rows_for_date(
    vehicle_positions_for_date: gpd.GeoDataFrame,
    service_date: str,
    shapes: gpd.GeoDataFrame,
    signals_projected: pd.Series,
    segments: pd.DataFrame,
    shape_key_map: dict[str, str],
    dwell_exclusion_distances: np.ndarray,
) -> list[dict]:
    """Project, smooth, and analyze every trip on a single service date."""
    distance_along_shape = project_vp_on_shape(
        vehicle_positions_for_date,
        shapes,
        shape_key_map,
        max_snap_distance=MAX_SNAP_DISTANCE_M,
        max_shape_jump=MAX_SHAPE_JUMP_M,
    )
    smoothed = smooth_distances_per_trip(
        vehicle_positions_for_date, distance_along_shape, freq_seconds=SMOOTH_FREQ_SECONDS
    )
    if smoothed.empty:
        return []
    speeds = compute_speeds_per_trip(smoothed, freq_seconds=SMOOTH_FREQ_SECONDS)
    smoothed_with_speed = smoothed.merge(
        speeds, on=["TRIP_KEY", "event_time_datetime"], how="inner"
    )

    rows = []
    for trip_key, trip in smoothed_with_speed.groupby("TRIP_KEY", sort=False):
        trip = trip.sort_values("event_time_datetime")
        trip_index = pd.DatetimeIndex(trip["event_time_datetime"])
        trajectory = pd.Series(
            trip["distance_along_shape_smoothed"].to_numpy(), index=trip_index
        )
        trip_speeds = pd.Series(trip["speed_m_per_s"].to_numpy(), index=trip_index)
        rows.extend(
            _segment_rows_for_trip(
                service_date,
                trip_key,
                trajectory,
                trip_speeds,
                signals_projected,
                segments,
                dwell_exclusion_distances,
            )
        )
    return rows


def build_trip_segment_table(
    vehicle_positions: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    signals_projected: pd.Series,
    segments: pd.DataFrame,
    shape_key_map: dict[str, str],
    dwell_exclusion_distances: np.ndarray,
) -> pd.DataFrame:
    """One row per (trip, segment) with raw travel time and stop quantities.

    `vehicle_positions` must carry a `service_date` column; trips are processed
    per date because TRIP_KEY is only unique within a date.
    """
    rows = []
    for service_date, positions_for_date in vehicle_positions.groupby(
        "service_date", sort=True
    ):
        rows.extend(
            _segment_rows_for_date(
                positions_for_date,
                service_date,
                shapes,
                signals_projected,
                segments,
                shape_key_map,
                dwell_exclusion_distances,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "service_date",
            "TRIP_KEY",
            "period",
            "segment_id",
            "observed_travel_time_s",
            "dwell_s",
            "uniform_stop_duration_s",
            "overflow_stop_duration_s",
            "has_signal_stop",
        ],
    )


def _night_percentile(
    trip_segment_table: pd.DataFrame,
    value_column: str,
    percentile: float,
    row_mask: pd.Series | None = None,
    name: str = "value",
) -> pd.Series:
    """Per-segment percentile of `value_column` over qualifying night rows."""
    night = trip_segment_table[trip_segment_table["period"] == "night"]
    if row_mask is not None:
        night = night[row_mask.loc[night.index]]
    grouped = night.groupby("segment_id")[value_column]
    result = grouped.quantile(percentile / 100)
    result[grouped.count() < MIN_BASELINE_TRIPS] = np.nan
    return result.rename(name)


def compute_free_flow_travel_times(trip_segment_table: pd.DataFrame) -> pd.Series:
    """T_ff per segment: FREE_FLOW_PERCENTILE-th pct of night travel times."""
    return _night_percentile(
        trip_segment_table,
        "observed_travel_time_s",
        FREE_FLOW_PERCENTILE,
        name="free_flow_travel_time_s",
    )


def compute_red_phase_caps(trip_segment_table: pd.DataFrame) -> pd.Series:
    """Uniform-delay cap per segment: RED_PHASE_PERCENTILE-th pct of night
    stop-bar (signal) stop durations. NaN (uncapped) where too few samples."""
    return _night_percentile(
        trip_segment_table,
        "uniform_stop_duration_s",
        RED_PHASE_PERCENTILE,
        row_mask=trip_segment_table["has_signal_stop"],
        name="red_phase_cap_s",
    )


def decompose_delays(
    trip_segment_table: pd.DataFrame,
    free_flow_travel_times: pd.Series,
    red_phase_caps: pd.Series,
    period: str = "day",
) -> pd.DataFrame:
    """Apply the red-phase cap and the residual decomposition to the chosen
    period's trip-segments (default daytime)."""
    rows = trip_segment_table[trip_segment_table["period"] == period].copy()
    rows["free_flow_travel_time_s"] = rows["segment_id"].map(free_flow_travel_times)
    rows["red_phase_cap_s"] = rows["segment_id"].map(red_phase_caps)

    cap = rows["red_phase_cap_s"].to_numpy()
    uniform_raw = rows["uniform_stop_duration_s"].to_numpy()
    has_cap = ~np.isnan(cap)

    capped_uniform = np.where(has_cap, np.minimum(uniform_raw, cap), uniform_raw)
    uniform_excess = np.where(has_cap, np.maximum(uniform_raw - cap, 0.0), 0.0)

    rows["uniform_delay_s"] = capped_uniform
    rows["overflow_delay_s"] = rows["overflow_stop_duration_s"].to_numpy() + uniform_excess
    rows["signal_delay_s"] = rows["uniform_delay_s"] + rows["overflow_delay_s"]
    rows["congestion_delay_s"] = (
        rows["observed_travel_time_s"]
        - rows["free_flow_travel_time_s"]
        - rows["dwell_s"]
        - rows["signal_delay_s"]
    ).clip(lower=0)
    rows["total_delay_s"] = rows["signal_delay_s"] + rows["congestion_delay_s"]
    return rows


DELAY_COMPONENTS = [
    "observed_travel_time_s",
    "uniform_delay_s",
    "overflow_delay_s",
    "signal_delay_s",
    "congestion_delay_s",
    "total_delay_s",
]


def segment_geometries(
    shapes: gpd.GeoDataFrame, shape_id: str, segments: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Sub-LineString of the shape for each segment (between its signal distances).

    Returns a GeoDataFrame (CRS of `shapes`) indexed like `segments`, for mapping.
    """
    from shapely.ops import substring

    line = shapes.set_index("shape_id")["geometry"][shape_id]
    geometries = [
        substring(line, segment.start_distance_m, segment.end_distance_m)
        for segment in segments.itertuples()
    ]
    return gpd.GeoDataFrame(
        segments.assign(geometry=geometries), geometry="geometry", crs=shapes.crs
    )


@dataclass
class SegmentDelayResult:
    """Tables produced by run_segment_delay_analysis (see module docstring)."""

    segments: pd.DataFrame
    trip_segment_table: pd.DataFrame
    daytime_delays: pd.DataFrame
    free_flow_travel_times: pd.Series
    red_phase_caps: pd.Series
    trip_counts: pd.DataFrame
    raw_travel_times: pd.DataFrame
    travel_time_matrix: pd.DataFrame
    component_stats: dict[str, pd.DataFrame]
    segment_summary: pd.DataFrame
    signals_projected: pd.Series
    signal_is_nearside: pd.Series


def _summarize(
    daytime_delays: pd.DataFrame,
    trip_segment_table: pd.DataFrame,
    free_flow_travel_times: pd.Series,
    red_phase_caps: pd.Series,
    segments: pd.DataFrame,
) -> dict:
    by_segment = daytime_delays.groupby("segment_id")

    night_trips = trip_segment_table[trip_segment_table["period"] == "night"]
    trip_counts = (
        pd.DataFrame({
            "n_day_trips": by_segment.size(),
            "n_signal_stop": daytime_delays[daytime_delays["has_signal_stop"]].groupby("segment_id").size(),
            "n_overflow": daytime_delays[daytime_delays["overflow_delay_s"] > 0].groupby("segment_id").size(),
            "n_congestion": daytime_delays[daytime_delays["congestion_delay_s"] > 0].groupby("segment_id").size(),
            "n_night_trips": night_trips.groupby("segment_id").size(),
        })
        .reindex(segments.index)
        .fillna(0)
        .astype(int)
    )

    raw_travel_times = (
        by_segment["observed_travel_time_s"].describe().join(free_flow_travel_times)
    )

    component_stats = {
        component: by_segment[component].describe() for component in DELAY_COMPONENTS
    }

    travel_time_matrix = daytime_delays.pivot_table(
        index=["service_date", "TRIP_KEY"],
        columns="segment_id",
        values="observed_travel_time_s",
    )

    mean_components = by_segment[
        ["uniform_delay_s", "overflow_delay_s", "signal_delay_s",
         "congestion_delay_s", "total_delay_s", "dwell_s"]
    ].mean().add_prefix("mean_")
    segment_summary = (
        segments.join(free_flow_travel_times)
        .join(red_phase_caps)
        .join(trip_counts)
        .join(mean_components)
    )

    return {
        "trip_counts": trip_counts,
        "raw_travel_times": raw_travel_times,
        "travel_time_matrix": travel_time_matrix,
        "component_stats": component_stats,
        "segment_summary": segment_summary,
    }


def run_segment_delay_analysis(
    vehicle_positions: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    signals: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    shape_id: str,
    shape_key_map: dict[str, str] = SHAPE_KEY_TO_SHAPE_ID_MAP,
) -> SegmentDelayResult:
    """Run the full §3.3 delay breakdown for one shape across all dates.

    Args:
        vehicle_positions: All trips for the shape, with a `service_date` column.
        shapes: GeoDataFrame from get_selected_shapes (shape_id + geometry).
        signals: Signal point GeoDataFrame from get_traffic_signals.
        stops: Stop GeoDataFrame with a boolean `nearside` column.
        shape_id: The shape to analyze (e.g. "shp-1-05").
        shape_key_map: ROUTE_ID -> shape_id mapping for projecting positions.
    """
    trip_shape = shapes.set_index("shape_id")["geometry"][[shape_id]]

    signals_projected = project_points_on_shape(signals, trip_shape, MAX_SNAP_DISTANCE_M)
    stops_projected = project_points_on_shape(stops, trip_shape, MAX_SNAP_DISTANCE_M)
    stops_nearside = stops.loc[stops_projected.index, "nearside"]

    signal_is_nearside = identify_nearside_signals(
        signals_projected, stops_projected, stops_nearside, NEARSIDE_STOP_DISTANCE_M
    )
    segments = build_segments(signals_projected, signal_is_nearside)

    # Stops near a bus stop or a near-side signal are dwell -> excluded from
    # signal/congestion attribution.
    dwell_exclusion_distances = np.concatenate([
        stops_projected.to_numpy(),
        signals_projected[signal_is_nearside].to_numpy(),
    ])

    trip_segment_table = build_trip_segment_table(
        vehicle_positions, shapes, signals_projected, segments, shape_key_map,
        dwell_exclusion_distances,
    )

    free_flow_travel_times = compute_free_flow_travel_times(trip_segment_table)
    red_phase_caps = compute_red_phase_caps(trip_segment_table)
    daytime_delays = decompose_delays(
        trip_segment_table, free_flow_travel_times, red_phase_caps, period="day"
    )

    summary = _summarize(
        daytime_delays, trip_segment_table, free_flow_travel_times, red_phase_caps, segments
    )

    return SegmentDelayResult(
        segments=segments,
        trip_segment_table=trip_segment_table,
        daytime_delays=daytime_delays,
        free_flow_travel_times=free_flow_travel_times,
        red_phase_caps=red_phase_caps,
        signals_projected=signals_projected,
        signal_is_nearside=signal_is_nearside,
        **summary,
    )


def example_trip_segments(daytime_delays: pd.DataFrame) -> pd.DataFrame:
    """One representative daytime (segment, trip) per delay flavor for plotting:
    uniform-only signal delay, overflow delay, and congestion-only.

    Returns a DataFrame with columns example, service_date, TRIP_KEY, segment_id,
    and the delay components, with the largest delay of each kind first.
    """
    selectors = {
        "signal (uniform)": (
            daytime_delays[
                (daytime_delays["uniform_delay_s"] > 0)
                & (daytime_delays["overflow_delay_s"] == 0)
            ],
            "uniform_delay_s",
        ),
        "overflow": (
            daytime_delays[daytime_delays["overflow_delay_s"] > 0],
            "overflow_delay_s",
        ),
        "congestion": (
            daytime_delays[
                (daytime_delays["signal_delay_s"] == 0)
                & (daytime_delays["congestion_delay_s"] > 0)
            ],
            "congestion_delay_s",
        ),
    }

    columns = [
        "service_date", "TRIP_KEY", "segment_id",
        "uniform_delay_s", "overflow_delay_s", "signal_delay_s", "congestion_delay_s",
    ]
    rows = []
    for label, (candidates, sort_column) in selectors.items():
        if candidates.empty:
            continue
        best = candidates.sort_values(sort_column, ascending=False).iloc[0]
        rows.append({"example": label, **{column: best[column] for column in columns}})
    return pd.DataFrame(rows, columns=["example", *columns])


@dataclass
class TripSegmentDetail:
    """One trip's trajectory through one segment, with classified stops."""

    trip_positions: pd.Series          # raw projected distance (m) vs datetime, sliced to segment
    segment: pd.Series                 # the segment row (signals, distances)
    segment_trajectory: pd.Series      # distance (m) vs datetime, sliced to the segment
    segment_speeds: pd.Series          # speed (m/s) vs datetime, sliced to the segment
    stops: pd.DataFrame                # in-segment stops: timestamps, duration, delay_type
    downstream_signal_distance_m: float


def analyze_trip_segment(
    vehicle_positions: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    signals: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    shape_id: str,
    service_date: str,
    trip_key,
    segment_id: str,
    shape_key_map: dict[str, str] = SHAPE_KEY_TO_SHAPE_ID_MAP,
) -> TripSegmentDetail:
    """Recompute one trip's smoothed trajectory through one segment and classify
    its stops, for plotting. Re-smooths just the selected trip."""
    trip_shape = shapes.set_index("shape_id")["geometry"][[shape_id]]
    signals_projected = project_points_on_shape(signals, trip_shape, MAX_SNAP_DISTANCE_M)
    stops_projected = project_points_on_shape(stops, trip_shape, MAX_SNAP_DISTANCE_M)
    stops_nearside = stops.loc[stops_projected.index, "nearside"]
    signal_is_nearside = identify_nearside_signals(
        signals_projected, stops_projected, stops_nearside, NEARSIDE_STOP_DISTANCE_M
    )
    segment = build_segments(signals_projected, signal_is_nearside).loc[segment_id]
    dwell_exclusion_distances = np.concatenate([
        stops_projected.to_numpy(),
        signals_projected[signal_is_nearside].to_numpy(),
    ])

    trip_positions = vehicle_positions[
        (vehicle_positions["service_date"] == service_date)
        & (vehicle_positions["TRIP_KEY"] == trip_key)
    ]
    distance_along_shape = project_vp_on_shape(
        trip_positions, shapes, shape_key_map,
        max_snap_distance=MAX_SNAP_DISTANCE_M, max_shape_jump=MAX_SHAPE_JUMP_M,
    )
    smoothed = smooth_distances_per_trip(
        trip_positions, distance_along_shape, freq_seconds=SMOOTH_FREQ_SECONDS
    )
    if smoothed.empty:
        raise ValueError(
            f"Trip {trip_key} on {service_date} has too few valid pings to smooth."
        )
    speeds_frame = compute_speeds_per_trip(smoothed, freq_seconds=SMOOTH_FREQ_SECONDS)
    merged = smoothed.merge(
        speeds_frame, on=["TRIP_KEY", "event_time_datetime"], how="inner"
    ).sort_values("event_time_datetime")

    trip_start = merged["event_time_datetime"].iloc[0]
    trip_index = pd.DatetimeIndex(merged["event_time_datetime"])
    trajectory = pd.Series(merged["distance_along_shape_smoothed"].to_numpy(), index=trip_index)
    speeds = pd.Series(merged["speed_m_per_s"].to_numpy(), index=trip_index)

    stopped_events = find_stopped_events(trajectory, speeds)
    speeds_by_second = _speeds_by_second(trajectory, speeds)
    in_segment = stopped_events[
        (stopped_events["end_distance_m"] > segment["start_distance_m"])
        & (stopped_events["end_distance_m"] <= segment["end_distance_m"])
    ].copy()

    if not in_segment.empty and dwell_exclusion_distances.size:
        distance_to_exclusion = np.abs(
            in_segment["end_distance_m"].to_numpy()[:, None]
            - dwell_exclusion_distances[None, :]
        ).min(axis=1)
        is_dwell = distance_to_exclusion <= DWELL_EXCLUSION_DISTANCE_M
    else:
        is_dwell = np.zeros(len(in_segment), dtype=bool)

    in_segment["delay_type"] = classify_segment_stops(
        in_segment, is_dwell, speeds_by_second, segment["end_distance_m"]
    ).to_numpy()
    in_segment["start_timestamp"] = trip_start + pd.to_timedelta(in_segment["start_time_s"], unit="s")
    in_segment["end_timestamp"] = trip_start + pd.to_timedelta(in_segment["end_time_s"], unit="s")

    in_segment_mask = (trajectory >= segment["start_distance_m"]) & (
        trajectory <= segment["end_distance_m"]
    )

    # Raw projected GPS pings (pre-smoothing) sliced to the segment, for overlay.
    raw_distance_along_shape = (
        pd.Series(
            distance_along_shape.to_numpy(),
            index=pd.DatetimeIndex(trip_positions["event_time_datetime"]),
        )
        .dropna()
        .sort_index()
    )
    raw_distance_along_shape = raw_distance_along_shape[
        ~raw_distance_along_shape.index.duplicated(keep="first")
    ]
    raw_points_in_segment = raw_distance_along_shape[
        (raw_distance_along_shape >= segment["start_distance_m"])
        & (raw_distance_along_shape <= segment["end_distance_m"])
    ]

    return TripSegmentDetail(
        trip_positions=raw_points_in_segment,
        segment=segment,
        segment_trajectory=trajectory[in_segment_mask],
        segment_speeds=speeds[in_segment_mask],
        stops=in_segment[
            ["start_timestamp", "end_timestamp", "duration_s", "end_distance_m", "delay_type"]
        ].reset_index(drop=True),
        downstream_signal_distance_m=segment["end_distance_m"],
    )
