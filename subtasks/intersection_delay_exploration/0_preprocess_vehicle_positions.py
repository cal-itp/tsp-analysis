"""Consolidate zipped VehicleState position dumps into a single geoparquet.

The intake folder holds one ``.txt.zip`` per vehicle, each spanning a single
service day, but the folder as a whole covers many dates and the filenames
encode the dump time, not the event date. Reading all of them on every analysis
run is slow, so this step reads them once, parses the local EVENT_TIME, and
writes a single geoparquet that the loaders can read directly. The service date
for each row is recovered downstream from the ``event_time_datetime`` column.

Run as a script to (re)build the consolidated parquet:

    uv run preprocess_vehicle_positions.py
"""

import sys
from pathlib import Path

# Import the flat modules that live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("src")))

import geopandas as gpd
import pandas as pd
from calitp_data_analysis.gcs_pandas import GCSPandas
from calitp_data_analysis.gcs_geopandas import GCSGeoPandas
from constants import (
    CA_NAD83_Albers,
    CULVER_CITY_VEHICLE_POSITIONS_FOLDER,
    CULVER_CITY_VEHICLE_POSITIONS_PATH,
)


def _read_one_zip(gcs: GCSPandas, path: str) -> pd.DataFrame:
    # The second row (index 1) is a column-type descriptor, not data.
    return gcs.read_csv(
        f"gs://{path}",
        compression="zip",
        skiprows=lambda row_index: row_index == 1,
        low_memory=False,
    )


def build_vehicle_positions(
    intake_folder: str = CULVER_CITY_VEHICLE_POSITIONS_FOLDER,
    output_path: str = CULVER_CITY_VEHICLE_POSITIONS_PATH,
) -> int:
    """Read every zipped VehicleState dump in ``intake_folder`` and write them as
    a single geoparquet to ``output_path``.

    The output is a GeoDataFrame in CA_NAD83_Albers with an ``event_time_datetime``
    column and point geometry built from LONGITUDE/LATITUDE. Rows whose EVENT_TIME
    cannot be parsed (e.g. status rows with no timestamp) or that lack
    LONGITUDE/LATITUDE are dropped. Returns the number of rows written.
    """
    gcs = GCSPandas()
    gcs_geo = GCSGeoPandas()
    paths = gcs.gcs_filesystem.glob(intake_folder + "*.txt.zip")

    all_positions = pd.concat(
        [_read_one_zip(gcs, path) for path in paths],
        ignore_index=True,
    )

    all_positions["event_time_datetime"] = pd.to_datetime(
        all_positions["EVENT_TIME"].astype(str),
        format=r"%y%m%d%H%M%S.0",
        errors="coerce",
    )
    all_positions = all_positions.dropna(subset=["event_time_datetime"])

    # A column read as all-NaN floats in some files but strings in others becomes
    # a mixed-type object column after concat, which pyarrow cannot serialize to
    # parquet. Normalize object columns to a nullable string dtype (numeric
    # columns are left untouched).
    object_columns = all_positions.select_dtypes(include="object").columns
    all_positions[object_columns] = all_positions[object_columns].astype("string")

    all_positions = (
        gpd.GeoDataFrame(
            all_positions,
            geometry=gpd.points_from_xy(
                all_positions.LONGITUDE, all_positions.LATITUDE
            ),
            crs="EPSG:4326",
        )
        .dropna(subset=["LONGITUDE", "LATITUDE"])
        .to_crs(CA_NAD83_Albers)
    )

    gcs_geo.geo_data_frame_to_parquet(all_positions, output_path)

    return len(all_positions)


if __name__ == "__main__":
    row_count = build_vehicle_positions()
    print(f"Wrote {row_count:,} rows to {CULVER_CITY_VEHICLE_POSITIONS_PATH}")
