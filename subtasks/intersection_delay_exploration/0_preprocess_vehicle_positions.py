"""Regroup zipped VehicleState position dumps by service date.

The intake folder holds one ``.txt.zip`` per vehicle, each spanning a single
service day, but the folder as a whole covers many dates and the filenames
encode the dump time, not the event date. Reading all of them on every analysis
run is slow, so this step reads them once, groups rows by the local EVENT_TIME
date, and writes one parquet per service date that the loaders can read directly.

Run as a script to (re)build the by-date folder:

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
    CULVER_CITY_VEHICLE_POSITIONS_BY_DATE_FOLDER,
)


def _read_one_zip(gcs: GCSPandas, path: str) -> pd.DataFrame:
    # The second row (index 1) is a column-type descriptor, not data.
    return gcs.read_csv(
        f"gs://{path}",
        compression="zip",
        skiprows=lambda row_index: row_index == 1,
        low_memory=False,
    )


def build_vehicle_positions_by_date(
    intake_folder: str = CULVER_CITY_VEHICLE_POSITIONS_FOLDER,
    output_folder: str = CULVER_CITY_VEHICLE_POSITIONS_BY_DATE_FOLDER,
) -> list[str]:
    """Read every zipped VehicleState dump in ``intake_folder``, group rows by
    their local EVENT_TIME date, and write one geoparquet per date to
    ``output_folder`` as ``vehicle_positions_<YYYY-MM-DD>.parquet``.

    Each file is a GeoDataFrame in CA_NAD83_Albers with an ``event_time_datetime``
    column and point geometry built from LONGITUDE/LATITUDE. Rows whose EVENT_TIME
    cannot be parsed (e.g. status rows with no timestamp) or that lack
    LONGITUDE/LATITUDE are dropped. Returns the service dates written, sorted.
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

    written_service_dates = []
    for service_date, positions_for_date in all_positions.groupby(
        all_positions["event_time_datetime"].dt.date
    ):
        print(type(positions_for_date))
        service_date_str = service_date.isoformat()
        output_path = (
            f"{output_folder}vehicle_positions_{service_date_str}.parquet"
        )
        gcs_geo.geo_data_frame_to_parquet(positions_for_date, output_path)
        written_service_dates.append(service_date_str)

    return sorted(written_service_dates)


if __name__ == "__main__":
    service_dates = build_vehicle_positions_by_date()
    print(f"Wrote {len(service_dates)} service dates:")
    for service_date in service_dates:
        print(f"  {service_date}")
