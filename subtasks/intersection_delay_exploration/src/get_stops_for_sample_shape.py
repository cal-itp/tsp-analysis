from constants import CULVER_CITY_FEED_KEY, SERVICE_DATE
from _data_loaders import get_stops_for_shape

stops = get_stops_for_shape(SERVICE_DATE, CULVER_CITY_FEED_KEY, "shp-1-05")
stops.to_crs("EPSG:4326").to_file("stops_shp-1-05.geojson", driver="GeoJSON")
print(f"Saved {len(stops)} stops to stops_shp-1-05.geojson")
