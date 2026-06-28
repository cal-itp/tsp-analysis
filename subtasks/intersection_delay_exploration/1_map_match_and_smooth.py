"""Map-match consolidated vehicle positions and precompute smoothed trips.

Reads the consolidated raw positions written by ``0_preprocess_vehicle_positions.py``
(so the slow zip-reading step isn't repeated when matching/smoothing changes),
projects each position onto its GTFS shape, and writes two parquets:

1. Matched pings (geoparquet): every position with its matched GTFS ``shape_id``
   and ``distance_along_shape``, so the projection isn't recomputed in each
   notebook.
2. Smoothed trips (parquet): per-trip LOCREG-PCHIP smoothed distance and speed on
   an even SMOOTH_FREQ_SECONDS time grid, tagged with ``service_date`` and the
   GTFS ``shape_id``.

Both steps run per service date because TRIP_KEY is only unique within a date
(and the shape-jump filter must compare pings within a single trip).

Run after ``0_preprocess_vehicle_positions.py``:

    uv run 0_map_match_and_smooth.py
"""

import datetime
import sys
from pathlib import Path

# Import the flat modules that live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("src")))

import geopandas as gpd
import pandas as pd
from calitp_data_analysis.gcs_pandas import GCSPandas
from calitp_data_analysis.gcs_geopandas import GCSGeoPandas
from src.constants import (
    CULVER_CITY_FEED_KEY,
    CULVER_CITY_MATCHED_VEHICLE_POSITIONS_PATH,
    CULVER_CITY_SMOOTHED_VEHICLE_POSITIONS_PATH,
    CULVER_CITY_VEHICLE_POSITIONS_PATH,
    MAX_SHAPE_JUMP_M,
    MAX_SNAP_DISTANCE_M,
    MAX_SPEED_M_S,
    SERVICE_DATE,
    SHAPE_KEY_TO_SHAPE_ID_MAP,
    SMOOTH_FREQ_SECONDS,
)
from src._data_loaders import get_selected_shapes
from src.match_shapes_vp import project_vp_on_shape
from src.smooth_trajectory import compute_speeds_per_trip, smooth_distances_per_trip


def build_matched_vehicle_positions(
    vehicle_positions: gpd.GeoDataFrame,
    output_path: str = CULVER_CITY_MATCHED_VEHICLE_POSITIONS_PATH,
    shape_key_map: dict[str, str] = SHAPE_KEY_TO_SHAPE_ID_MAP,
) -> gpd.GeoDataFrame:
    """Project every position onto its GTFS shape and write the result to ``output_path``.

    Adds a ``shape_id`` column (the GTFS shape each position's ROUTE_ID maps to via
    ``shape_key_map``) and a ``distance_along_shape`` column (meters along that
    shape, NaN where the ping is beyond MAX_SNAP_DISTANCE_M / MAX_SHAPE_JUMP_M).
    Positions whose ROUTE_ID has no shape in ``shape_key_map`` are dropped.

    Projection is done per service date so the shape-jump filter only ever
    compares pings within a single trip (TRIP_KEY is unique within a date).
    Returns the written GeoDataFrame.
    """
    gcs_geo = GCSGeoPandas()

    # Shapes are schedule data and stable across the date range, so fetch them
    # once for the default service date.
    shapes = get_selected_shapes(
        SERVICE_DATE, CULVER_CITY_FEED_KEY, list(shape_key_map.values())
    )

    per_date_matched = []
    for service_date, positions_for_date in vehicle_positions.groupby(
        vehicle_positions["event_time_datetime"].dt.date, sort=True
    ):
        print(f"matching pings for {service_date}")
        shape_ids = positions_for_date["ROUTE_ID"].str.strip().map(shape_key_map)
        distance_along_shape = project_vp_on_shape(
            positions_for_date,
            shapes,
            shape_key_map,
            max_snap_distance=MAX_SNAP_DISTANCE_M,
            max_shape_jump=MAX_SHAPE_JUMP_M,
            max_speed_between_pings=MAX_SPEED_M_S
        )
        matched_for_date = positions_for_date.assign(
            shape_id=shape_ids.astype("string"),
            distance_along_shape=distance_along_shape,
        )
        per_date_matched.append(matched_for_date[matched_for_date["shape_id"].notna()])

    matched = pd.concat(per_date_matched)

    gcs_geo.geo_data_frame_to_parquet(matched, output_path)

    return matched


def build_smoothed_vehicle_positions(
    matched_positions: gpd.GeoDataFrame,
    output_path: str = CULVER_CITY_SMOOTHED_VEHICLE_POSITIONS_PATH,
) -> pd.DataFrame:
    """Smooth every trip's trajectory and write the per-trip speeds to ``output_path``.

    Reuses the ``distance_along_shape`` already computed by
    ``build_matched_vehicle_positions`` (no re-projection), smooths each trip's
    distance with LOCREG-PCHIP, and computes per-trip speeds.

    Trips are processed per service date because TRIP_KEY is only unique within a
    date. Returns the written DataFrame with columns service_date, shape_id,
    TRIP_KEY, event_time_datetime, distance_along_shape_smoothed, speed_m_per_s on
    an even SMOOTH_FREQ_SECONDS time grid within each trip.
    """
    gcs = GCSPandas()

    per_date_smoothed = []
    for service_date, positions_for_date in matched_positions.groupby(
        matched_positions["event_time_datetime"].dt.date, sort=True
    ):
        print(f"smoothing trips for {service_date}")
        vp_smoothed = smooth_distances_per_trip(
            positions_for_date,
            positions_for_date["distance_along_shape"],
            freq_seconds=SMOOTH_FREQ_SECONDS,
        )
        if vp_smoothed.empty:
            continue
        vp_speeds = compute_speeds_per_trip(vp_smoothed, freq_seconds=SMOOTH_FREQ_SECONDS)
        smoothed_with_speed = vp_smoothed.merge(
            vp_speeds, on=["TRIP_KEY", "event_time_datetime"], how="left" # TODO: note changed inner merge to outer - wonder if this was dropping pings?
        )
        # Each trip maps to one shape; attach it for shape-based filtering.
        trip_shape_ids = positions_for_date.groupby("TRIP_KEY")["shape_id"].first()
        smoothed_with_speed.insert(
            0, "shape_id", smoothed_with_speed["TRIP_KEY"].map(trip_shape_ids)
        )
        smoothed_with_speed.insert(0, "service_date", service_date.isoformat())
        per_date_smoothed.append(smoothed_with_speed)

    all_smoothed = pd.concat(per_date_smoothed, ignore_index=True)

    gcs.data_frame_to_parquet(all_smoothed, output_path)

    return all_smoothed


if __name__ == "__main__":
    print("reading")
    vehicle_positions = GCSGeoPandas().read_parquet(CULVER_CITY_VEHICLE_POSITIONS_PATH)
    print("read")
    vp_check_indices = vehicle_positions.loc[
        (vehicle_positions["TRIP_KEY"] == "986.0")
        & (vehicle_positions["EVENT_TIME_UTC"] >= 260204185050.0)
        & (vehicle_positions["EVENT_TIME_UTC"] <= 260204185727.0)
        & (vehicle_positions["EVENT_TIME_UTC"].notna())
    ]
    print("vp check len", len(vp_check_indices))
    matched = build_matched_vehicle_positions(vehicle_positions)
    print(
        f"Wrote {len(matched):,} matched pings to "
        f"{CULVER_CITY_MATCHED_VEHICLE_POSITIONS_PATH}"
    )

    smoothed = build_smoothed_vehicle_positions(matched)
    trip_count = len(smoothed.groupby(["service_date", "TRIP_KEY"]))
    print(
        f"Wrote {len(smoothed):,} smoothed samples across "
        f"{trip_count:,} trips ({smoothed['service_date'].nunique()} service dates) to "
        f"{CULVER_CITY_SMOOTHED_VEHICLE_POSITIONS_PATH}"
    )
