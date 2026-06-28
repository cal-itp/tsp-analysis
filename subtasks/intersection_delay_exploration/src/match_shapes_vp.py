import numpy as np
import pandas as pd
import geopandas as gpd
from calitp_data_analysis import sql
from calitp_data_analysis.geography_utils import make_routes_gdf
from .constants import CA_NAD83_Albers, CULVER_CITY_FEED_KEY, SERVICE_DATE


def project_vp_on_shape(
    vehicle_positions: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    shape_key_map: dict[str, str],
    max_snap_distance: float,
    max_shape_jump: float,
) -> pd.Series:
    """Return each vehicle position's distance (meters) along its respective shape.

    Args:
        vehicle_positions: GeoDataFrame from get_culver_city_vehicle_positions.
        shapes: GeoDataFrame from get_selected_shapes.
        shape_key_map: Mapping from ROUTE_ID values to shape_id values.
        max_snap_distance: Points whose distance from the projected point exceeds
            this threshold (meters) are set to NaN.
        max_shape_jump: Points where the distance-along-shape jumps more than this
            threshold (meters) from the previous ping within the same trip are set to NaN.

    Returns:
        Float Series indexed identically to vehicle_positions, where each value
        is the distance in meters from the start of the matched shape LineString
        to the projected point. Rows whose ROUTE_ID has no matching shape are NaN.
    """
    # Build a shape_id -> LineString lookup for fast access per group
    shape_geom = shapes.set_index("shape_id")["geometry"]

    # Map each vehicle position's ROUTE_ID to the corresponding GTFS shape_id
    vp_shape_ids = vehicle_positions["ROUTE_ID"].str.strip().map(shape_key_map)

    # Default all rows to NaN; filled in below for rows with a matched shape
    result = pd.Series(
        float("nan"), index=vehicle_positions.index, name="distance_along_shape"
    )

    for shape_id, group in vehicle_positions.groupby(vp_shape_ids):
        line = shape_geom.get(shape_id)
        if line is None:
            continue

        # Project each point onto the line to get distance from the shape's start
        projected = group.geometry.map(line.project)

        # Compute the distance from each original point to its projected counterpart
        # by interpolating back to a geometry and measuring the gap
        snap_dist = group.geometry.map(
            lambda pt: pt.distance(line.interpolate(line.project(pt)))
        )
        # Null out points that are too far from the shape to be reliable
        projected[snap_dist > max_snap_distance] = float("nan")

        # Null out points where consecutive projected distances jump too far within a trip
        jump = group.assign(_proj=projected).groupby("TRIP_KEY")["_proj"].diff()
        jump_indices = (jump.abs() > max_shape_jump)
        last_jump = ()
        #TODO: there are instances where there are multiple jumped positions in a row, which this doesn't catch
        # we're trying to lookahead to see where those are, to make sure we get all of them
        # i'm really confused about how to do that without ugly code

        projected[jump_indices] = float("nan")

        result.loc[group.index] = projected

    return result


def project_points_on_shape(
    points: gpd.GeoDataFrame,
    shape: gpd.GeoDataFrame,
    max_snap_distance: float,
) -> pd.Series:
    """Return each nearby point's distance (meters) along shape from its start.

    points farther than max_snap_distance from the shape are excluded before
    projection. Returns a Series indexed like the filtered subset of points.

    Args:
        points: GeoDataFrame containing points.
        shape: GeoDataFrame containing single linestring to project onto - must be a geographic CRS.
        max_snap_distance: Maximum distance (meters) from the shape to include a point.

    Returns:
        Float Series indexed like the filtered subset of points.
    """
    projected_shape_geoseries = shape.to_crs(shape.crs)
    projected_shape_geom = projected_shape_geoseries.iloc[0]
    nearby_points = points[points.geometry.map(lambda pt: pt.distance(projected_shape_geom)) <= max_snap_distance]
    return nearby_points.geometry.map(projected_shape_geom.project)


def distances_to_crossing_times(
    distances_along_shape: pd.Series,
    smoothed_trip_distances: pd.Series,
) -> pd.Series:
    """Look up the time the bus crossed each input distance along its shape.

    Linearly interpolates against a single trip's smoothed distance-over-time
    trajectory. Input distances outside the trip's distance range are dropped.

    Args:
        distances_along_shape: Distances (meters) along the trip's shape to
            look up crossing times for, e.g. signal or stop projections.
        smoothed_trip_distances: Distance-along-shape (meters) for a single
            trip, indexed by event_time_datetime and monotonically increasing.

    Returns:
        Series of crossing timestamps, indexed like the in-range subset of
        distances_along_shape.
    """
    trip_distances_m = smoothed_trip_distances.to_numpy()
    trip_times_ns = smoothed_trip_distances.index.astype("int64").to_numpy()
    in_trip_range = (distances_along_shape >= trip_distances_m[0]) & (
        distances_along_shape <= trip_distances_m[-1]
    )
    valid_distances = distances_along_shape[in_trip_range]
    crossing_times_ns = np.interp(
        valid_distances.to_numpy(), trip_distances_m, trip_times_ns
    )
    return pd.Series(
        pd.to_datetime(crossing_times_ns.astype("int64"), unit="ns"),
        index=valid_distances.index,
    )
