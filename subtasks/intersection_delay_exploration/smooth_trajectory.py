import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def _tricube(u: np.ndarray) -> np.ndarray:
    """Cleveland's tricube kernel: (1 - |u|^3)^3 for |u| < 1, else 0."""
    abs_u = np.abs(u)
    return np.where(abs_u < 1, (1 - abs_u**3) ** 3, 0.0)


def _locreg(times: np.ndarray, values: np.ndarray, k: int, degree: int) -> np.ndarray:
    """Locally-weighted regression with tricube kernel and k-NN bandwidth.

    For each input time t_i, fits a weighted polynomial of `degree` to the k
    closest points (by |t - t_i|), weighted by the tricube kernel scaled to the
    distance of the k-th neighbor. Returns the polynomial's prediction at t_i.
    """
    n = len(times)
    smoothed = np.empty(n)
    for i in range(n):
        dists = np.abs(times - times[i])
        # k-th smallest distance defines the local bandwidth h_i
        h = np.partition(dists, k - 1)[k - 1]
        if h == 0:
            # Degenerate (e.g. duplicate timestamps): use uniform weights
            # over the k closest points.
            idx = np.argpartition(dists, k - 1)[:k]
            t_local = times[idx]
            v_local = values[idx]
            sqrt_w = np.ones(k)
        else:
            weights = _tricube(dists / h)
            mask = weights > 0
            t_local = times[mask]
            v_local = values[mask]
            sqrt_w = np.sqrt(weights[mask])
        coefs = np.polyfit(t_local, v_local, degree, w=sqrt_w)
        smoothed[i] = np.polyval(coefs, times[i])
    return smoothed


def smooth_trajectory(
    times: np.ndarray,
    distances: np.ndarray,
    k: int = 20,
    degree: int = 3,
) -> PchipInterpolator:
    """Smooth a (time, distance) trajectory using LOCREG-PCHIP.

    Implements Locreg-PCHIP from Huang et al. 2023 (arXiv:2305.15545):
        1. Locally-weighted regression with the tricube kernel and a k-nearest-
           neighbor bandwidth produces a smoothed distance estimate at each
           input time.
        2. Monotonicity is enforced by clamping any backward step to the
           running max.
        3. A monotonic piecewise cubic Hermite interpolant (PCHIP) is fit
           through the cleaned values, giving a continuous, monotonic, once-
           differentiable trajectory.

    Args:
        times: Strictly increasing 1D array of timestamps in seconds.
        distances: 1D array of distance-along-shape values, same length as
            times. Both inputs must be free of NaN.
        k: Bandwidth as number of nearest neighbors used by the local
            regression. Defaults to 20 (the value selected in the paper).
        degree: Local polynomial degree. Defaults to 3.

    Returns:
        A PchipInterpolator f such that f(t) is the smoothed distance at time
        t. Speed at time t is f.derivative()(t).
    """
    times = np.asarray(times, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if times.shape != distances.shape:
        raise ValueError("times and distances must have the same shape")
    if times.ndim != 1:
        raise ValueError("times and distances must be 1D")
    if len(times) < k:
        raise ValueError(f"need at least k={k} points, got {len(times)}")

    smoothed = _locreg(times, distances, k=k, degree=degree)

    # Monotonicity sweep: backward steps clamp to the running max
    # TODO: investigate unrealistic speeds (>40 mph) reaching compute_speeds_per_trip.
    # Suspected causes, in order of likely contribution:
    #   1. THIS monotonicity sweep is the prime suspect. It clamps backward
    #      LOCREG noise but preserves forward noise as a permanent positive
    #      offset via the running max, so each upward step becomes a fast
    #      spike in d/dt. Asymmetric handling -> systematic upward bias on
    #      speeds. Consider replacing with isotonic regression (symmetric)
    #      or dropping the sweep entirely.
    #   2. PCHIP across sparse-ping gaps. project_vp_on_shape nulls pings
    #      that fail MAX_SNAP_DISTANCE_M / MAX_SHAPE_JUMP_M; in urban
    #      canyons or complex intersections this can leave long gaps that
    #      PCHIP interpolates monotonically. If the bus was stopped during
    #      part of the gap, the moving portion gets implicitly redistributed
    #      and apparent speed climbs.
    #   3. Edge effects. LOCREG kernel is asymmetric at trip endpoints, and
    #      np.gradient in compute_speeds_per_trip is one-sided at the first
    #      and last sample. Speeds at row 0 / row -1 of a trip are noisier.
    #   4. Snap jitter that survives the 20m filter. LOCREG dampens it by
    #      ~sqrt(k) ≈ 4.5x, leaving ~4m residual amplitude — ~9 mph noise
    #      floor on bare differencing at 1s freq before #1 amplifies it.
    #   5. Genuine peaks. Washington Blvd posted speed is mostly 35 mph
    #      with 40 mph stretches; brief 40-45 mph readings are plausible.
    # Quickest diagnostic: plot per-trip speeds against raw projections and
    # the dropped-ping mask. If >40 mph samples line up with running-max
    # ticks, that confirms (1); if with dropped-ping stretches, (2); if at
    # trip start/end, (3). Cheap interim mitigation: post-hoc clamp speeds
    # to a configurable ceiling (e.g. 50 mph).
    np.maximum.accumulate(smoothed, out=smoothed)

    return PchipInterpolator(times, smoothed)


def smooth_distances_per_trip(
    vehicle_positions: pd.DataFrame,
    distance_along_shape: pd.Series,
    freq_seconds: float,
    k: int = 20,
    degree: int = 3,
) -> pd.DataFrame:
    """Apply LOCREG-PCHIP smoothing per trip on an evenly-spaced time grid.

    For each TRIP_KEY group: drops NaN distances and duplicate timestamps,
    fits a smoothed trajectory with smooth_trajectory(), then evaluates it
    on an evenly-spaced grid spanning the trip's first to last valid ping.
    Trips with fewer than k valid pings are skipped.

    Args:
        vehicle_positions: DataFrame with TRIP_KEY and event_time_datetime
            columns, as produced by get_culver_city_vehicle_positions.
        distance_along_shape: Series of distances along shape (meters), indexed
            identically to vehicle_positions, as produced by project_vp_on_shape.
        freq_seconds: Spacing (in seconds) between consecutive output samples
            within each trip.
        k: Bandwidth as number of nearest neighbors used by the local
            regression. Defaults to 20.
        degree: Local polynomial degree. Defaults to 3.

    Returns:
        DataFrame with columns TRIP_KEY, event_time_datetime, and
        distance_along_shape_smoothed. Within each trip the timestamps are
        evenly spaced freq_seconds apart, starting at the first valid ping
        and not exceeding the last.
    """
    per_trip_smoothed_dataframes = []

    for trip_key, group in vehicle_positions.groupby("TRIP_KEY", sort=False):
        trip = (
            group.assign(_d=distance_along_shape.loc[group.index])
            .dropna(subset=["_d"])
            .sort_values("event_time_datetime")
        )
        # PchipInterpolator requires strictly increasing input
        trip = trip[~trip["event_time_datetime"].duplicated(keep="first")]
        if len(trip) < k:
            continue

        t0 = trip["event_time_datetime"].iloc[0]
        times = (trip["event_time_datetime"] - t0).dt.total_seconds().to_numpy()
        distances = trip["_d"].to_numpy()

        f = smooth_trajectory(times, distances, k=k, degree=degree)

        sample_times = np.arange(0.0, times[-1] + freq_seconds, freq_seconds)
        sample_times = sample_times[sample_times <= times[-1]]

        per_trip_smoothed_dataframes.append(
            pd.DataFrame(
                {
                    "TRIP_KEY": trip_key,
                    "event_time_datetime": t0 + pd.to_timedelta(sample_times, unit="s"),
                    "distance_along_shape_smoothed": f(sample_times),
                }
            )
        )

    if not per_trip_smoothed_dataframes:
        return pd.DataFrame(
            columns=["TRIP_KEY", "event_time_datetime", "distance_along_shape_smoothed"]
        )

    return pd.concat(per_trip_smoothed_dataframes, ignore_index=True)


def compute_speeds_per_trip(
    smoothed: pd.DataFrame,
    freq_seconds: float,
) -> pd.DataFrame:
    """Compute per-trip speeds from a smoothed distance-along-shape DataFrame.

    Uses centered finite differences (np.gradient) on the evenly-spaced
    smoothed distance samples, so the output grid matches the input row-for-row.

    Args:
        smoothed: DataFrame produced by smooth_distances_per_trip, with columns
            TRIP_KEY, event_time_datetime, distance_along_shape_smoothed and
            uniform freq_seconds spacing within each TRIP_KEY group.
        freq_seconds: Sample spacing (seconds) used to build `smoothed`. Must
            match the value passed to smooth_distances_per_trip.

    Returns:
        DataFrame with columns TRIP_KEY, event_time_datetime, and speed_m_per_s,
        aligned 1:1 with `smoothed`.

    Notes:
        TODO: fact-check
        Recommended minimum freq_seconds is ~1 second. The LOCREG-PCHIP
        smoothing already low-pass-filters the trajectory across k≈20 pings
        (~50-100s of real time), so sampling speeds finer than ~1s is
        oversampling rather than extra resolution, and individual samples
        become dominated by GPS quantization (~5m / sqrt(k) ≈ 1m residual
        noise) and PCHIP segment-boundary wobble. Use 2-5s for cleaner speed
        traces when fine time resolution isn't needed.
    """
    per_trip_speed_dataframes = []
    for trip_key, group in smoothed.groupby("TRIP_KEY", sort=False):
        group = group.sort_values("event_time_datetime")
        distances = group["distance_along_shape_smoothed"].to_numpy()
        per_trip_speed_dataframes.append(
            pd.DataFrame(
                {
                    "TRIP_KEY": trip_key,
                    "event_time_datetime": group["event_time_datetime"].to_numpy(),
                    "speed_m_per_s": np.gradient(distances, freq_seconds),
                }
            )
        )

    if not per_trip_speed_dataframes:
        return pd.DataFrame(
            columns=["TRIP_KEY", "event_time_datetime", "speed_m_per_s"]
        )

    return pd.concat(per_trip_speed_dataframes, ignore_index=True)
