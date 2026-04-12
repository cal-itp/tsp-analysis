import pandas as pd
import geopandas as gpd
from calitp_data_analysis import sql
from calitp_data_analysis.geography_utils import make_routes_gdf
from constants import CA_NAD83_Albers, CULVER_CITY_FEED_KEY, SERVICE_DATE


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
        jump = group.assign(_proj=projected).groupby("TRIP_KEY")["_proj"].diff().abs()
        projected[jump > max_shape_jump] = float("nan")

        result.loc[group.index] = projected

    return result
