from .constants import CULVER_CITY_FEED_KEY, DATA_DIR, SERVICE_DATE
from ._data_loaders import get_stops_for_shape

output_path = DATA_DIR.joinpath("stops_shp-1-05.geojson")
stops = get_stops_for_shape(SERVICE_DATE, CULVER_CITY_FEED_KEY, "shp-1-05")
stops.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
print(f"Saved {len(stops)} stops to {output_path}")
