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
    np.maximum.accumulate(smoothed, out=smoothed)

    return PchipInterpolator(times, smoothed)


def smooth_distances_per_trip(
    vehicle_positions: pd.DataFrame,
    distance_along_shape: pd.Series,
    k: int = 20,
    degree: int = 3,
) -> pd.Series:
    """Apply LOCREG-PCHIP smoothing per trip to a distance-along-shape series.

    For each TRIP_KEY group: drops NaN distances and duplicate timestamps,
    converts event_time_datetime to seconds since the first valid ping, fits
    a smoothed trajectory with smooth_trajectory(), and evaluates it at the
    kept ping times. Trips with fewer than k valid pings are skipped (NaN).

    Args:
        vehicle_positions: DataFrame with TRIP_KEY and event_time_datetime
            columns, as produced by get_culver_city_vehicle_positions.
        distance_along_shape: Series of distances along shape (meters), indexed
            identically to vehicle_positions, as produced by project_vp_on_shape.
        k: Bandwidth as number of nearest neighbors used by the local
            regression. Defaults to 20.
        degree: Local polynomial degree. Defaults to 3.

    Returns:
        Float Series indexed identically to vehicle_positions, where each value
        is the smoothed distance along shape (meters). Pings dropped for any
        reason (NaN input distance, duplicate timestamp, trip too short) are NaN.
    """
    result = pd.Series(
        float("nan"),
        index=vehicle_positions.index,
        name="distance_along_shape_smoothed",
    )

    for _, group in vehicle_positions.groupby("TRIP_KEY", sort=False):
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
        result.loc[trip.index] = f(times)

    return result
