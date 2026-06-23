from pathlib import Path

# Local reference data lives in <subtask>/data; resolve it from this file's
# location so paths work regardless of the current working directory.
DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")

CA_NAD83_Albers = "EPSG:3310"
CULVER_CITY_SIGNAL_LOCATIONS_PATH = str(DATA_DIR.joinpath("culver_signals.geojson"))
CULVER_CITY_FEED_KEY = "90a34032bdea10f106a3922133c46444"
# Intake folder of zipped VehicleState position dumps. Each zip covers one
# vehicle's events and the folder spans many service dates (2026-01-31 ..
# 2026-02-16); the service date is derived from EVENT_TIME content, not the
# filename. Run preprocess_vehicle_positions.py to regroup these by date.
CULVER_CITY_VEHICLE_POSITIONS_FOLDER = (
    "gs://calitp-analytics-data/data-analyses/tsp-analysis/culver_city_2026-02_partial/"
)
# Output of the preprocessing step: a single geoparquet covering all service
# dates, read directly by the loaders. The service date for each row is derived
# from its event_time_datetime column.
CULVER_CITY_VEHICLE_POSITIONS_PATH = (
    "gs://calitp-analytics-data/data-analyses/tsp-analysis/processed/culver_city_vehicle_positions.parquet"
)
# Output of the map-matching step: every consolidated position with its matched
# GTFS shape_id and distance_along_shape, so the projection isn't recomputed in
# each notebook. A geoparquet (carries geometry).
CULVER_CITY_MATCHED_VEHICLE_POSITIONS_PATH = (
    "gs://calitp-analytics-data/data-analyses/tsp-analysis/processed/culver_city_matched_vehicle_positions.parquet"
)
# Output of the smoothing step: per-trip smoothed distance-along-shape and speed,
# on an even SMOOTH_FREQ_SECONDS time grid. Columns: service_date, shape_id,
# TRIP_KEY, event_time_datetime, distance_along_shape_smoothed, speed_m_per_s.
CULVER_CITY_SMOOTHED_VEHICLE_POSITIONS_PATH = (
    "gs://calitp-analytics-data/data-analyses/tsp-analysis/processed/culver_city_smoothed_vehicle_positions.parquet"
)
# Default service date to analyze; override in notebooks to any date in the folder.
SERVICE_DATE = "2026-02-03"
MAX_SNAP_DISTANCE_M = 20
MAX_SHAPE_JUMP_M = 200
SMOOTH_FREQ_SECONDS = 1.0  # smoothing/speed sample spacing

SHAPE_KEY_TO_SHAPE_ID_MAP = {
    "105": "shp-1-05",
    "104": "shp-1-04",
}