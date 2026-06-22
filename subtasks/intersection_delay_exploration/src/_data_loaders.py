import geopandas as gpd
import pandas as pd
from calitp_data_analysis.gcs_geopandas import GCSGeoPandas
from calitp_data_analysis import sql
from calitp_data_analysis.geography_utils import make_routes_gdf
from constants import (
    CA_NAD83_Albers,
    CULVER_CITY_SIGNAL_LOCATIONS_PATH,
    CULVER_CITY_VEHICLE_POSITIONS_PATH,
)
from functools import cache


@cache
def _gcs_geopandas():
    return GCSGeoPandas()


@cache
def _all_vehicle_positions() -> gpd.GeoDataFrame:
    """Read the consolidated vehicle-positions geoparquet once (cached for the
    session). Covers all service dates; callers filter by event_time_datetime.
    """
    return _gcs_geopandas().read_parquet(CULVER_CITY_VEHICLE_POSITIONS_PATH)


def list_available_service_dates() -> list[str]:
    """Return the sorted service dates (YYYY-MM-DD) present in the consolidated
    vehicle-positions geoparquet, derived from event_time_datetime.
    """
    dates = _all_vehicle_positions()["event_time_datetime"].dt.date.unique()
    return sorted(service_date.isoformat() for service_date in dates)


def get_culver_city_vehicle_positions(
    route_ids: list[str],
    service_date: str,
) -> gpd.GeoDataFrame:
    """Load and minimally process Culver City CAD/AVL vehicle positions.

    Reads the consolidated geoparquet written by preprocess_vehicle_positions.py
    (a single file rather than every zipped dump) and filters to ``service_date``.
    Geometry, CRS (CA_NAD83_Albers), and the EVENT_TIME parse are produced there.

    Args:
        route_ids: ROUTE_ID values (whitespace-stripped) to keep, e.g. shape keys.
        service_date: Date string in YYYY-MM-DD format; rows whose
            event_time_datetime falls on this date are kept.

    Returns a GeoDataFrame (CRS CA_NAD83_Albers) with:
    - event_time_datetime: parsed datetime from EVENT_TIME
    - time_difference: seconds since previous ping within each TRIP_KEY
    - geometry: point from LONGITUDE/LATITUDE
    filtered to the requested route_ids.
    """
    service_date_parsed = pd.Timestamp(service_date).date()
    gdf = _all_vehicle_positions()
    gdf = gdf.loc[gdf["event_time_datetime"].dt.date == service_date_parsed]

    gdf = gdf.sort_values(["TRIP_KEY", "event_time_datetime"], ascending=True)

    gdf["time_difference"] = (
        gdf.groupby("TRIP_KEY")["event_time_datetime"].diff().dt.total_seconds()
    )

    gdf = gdf.loc[gdf["ROUTE_ID"].str.strip().isin(route_ids)]

    return gdf

def get_selected_shapes(service_date: str, feed_key: str, shape_ids: list[str]) -> gpd.GeoDataFrame:
    """Fetch all GTFS shapes for a given service_date and feed_key from the warehouse.

    Args:
        service_date: Date string in YYYY-MM-DD format.
        feed_key: Feed key from fct_schedule_feed_downloads.
        shape_ids: Shape ids from the gtfs schedule 

    Returns a GeoDataFrame with one row per shape and a LineString geometry column.
    """ 
    shapes_sql = ", ".join(f"'{n}'" for n in shape_ids)
    df = sql.query_sql(
        f"""
        select *
        from `cal-itp-data-infra.mart_gtfs.fct_daily_scheduled_shapes`
        where service_date = '{service_date}'
          and feed_key = '{feed_key}'
          and shape_id in ({shapes_sql})
        """
    )
    return make_routes_gdf(df, crs=CA_NAD83_Albers)


def get_stops_for_shape(service_date: str, feed_key: str, shape_id: str) -> gpd.GeoDataFrame:
    """Fetch stops served by a given shape on a service date.

    Finds all trips in fct_scheduled_trips with the given service_date, feed_key,
    and shape_id, then returns the distinct stops those trips serve.

    Args:
        service_date: Date string in YYYY-MM-DD format.
        feed_key: Feed key from fct_schedule_feed_downloads.
        shape_id: Shape ID from the GTFS schedule.

    Returns all stops served by the route associated with shape_id on that
    service date, across all shapes for that route. Callers should apply a
    geospatial filter to restrict to stops along the specific shape.

    Returns a GeoDataFrame with columns: stop_id, stop_name, geometry (Point, EPSG:3310).
    No stop_sequence is available; order by projection onto the shape if needed.
    """
    df = sql.query_sql(
        f"""
        with trip_info as (
            select route_id
            from `cal-itp-data-infra.mart_gtfs.fct_scheduled_trips`
            where service_date = '{service_date}'
              and feed_key = '{feed_key}'
              and shape_id = '{shape_id}'
            limit 1
        )
        select
            scheduled_stops.stop_id,
            scheduled_stops.stop_name,
            scheduled_stops.pt_geom
        from trip_info
        inner join `cal-itp-data-infra.mart_gtfs.fct_daily_scheduled_stops` scheduled_stops
            on scheduled_stops.service_date = '{service_date}'
            and scheduled_stops.feed_key = '{feed_key}'
            and trip_info.route_id in unnest(scheduled_stops.route_id_array)
        """
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["pt_geom"]),
        crs="EPSG:4326",
    ).to_crs(CA_NAD83_Albers)
    return gdf[["stop_id", "stop_name", "geometry"]]


def get_traffic_signals(
    path: str = CULVER_CITY_SIGNAL_LOCATIONS_PATH,
) -> gpd.GeoDataFrame:
    """Load traffic signal locations from a GeoJSON file.

    Returns a GeoDataFrame in EPSG:3310 with all source fields plus an id column
    reflecting row order.
    """
    gdf = gpd.read_file(path).to_crs(CA_NAD83_Albers)
    gdf["id"] = range(len(gdf))
    return gdf