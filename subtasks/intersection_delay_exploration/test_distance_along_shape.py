import geopandas as gpd
import matplotlib.pyplot as plt

from constants import CA_NAD83_Albers, CULVER_CITY_FEED_KEY, MAX_SHAPE_JUMP_M, MAX_SNAP_DISTANCE_M, SERVICE_DATE, SHAPE_KEY_TO_SHAPE_ID_MAP
from _data_loaders import get_culver_city_vehicle_positions, get_selected_shapes, get_traffic_signals
from match_shapes_vp import project_vp_on_shape, project_signals_on_shape
from smooth_trajectory import smooth_distances_per_trip

route_ids = list(SHAPE_KEY_TO_SHAPE_ID_MAP.keys())
shape_ids = list(SHAPE_KEY_TO_SHAPE_ID_MAP.values())

print("Loading vehicle positions...")
vp = get_culver_city_vehicle_positions(route_ids)

print("Loading shapes...")
shapes = get_selected_shapes(SERVICE_DATE, CULVER_CITY_FEED_KEY, shape_ids)

print("Loading traffic signals...")
signals = get_traffic_signals()

print("Loading stops...")
stops = gpd.read_file("stops_shp-1-05.geojson").to_crs(CA_NAD83_Albers)

print("Projecting onto shapes...")
vp["distance_along_shape"] = project_vp_on_shape(vp, shapes, SHAPE_KEY_TO_SHAPE_ID_MAP, max_snap_distance=MAX_SNAP_DISTANCE_M, max_shape_jump=MAX_SHAPE_JUMP_M)

print("Smoothing trajectories...")
vp["distance_along_shape_smoothed"] = smooth_distances_per_trip(vp, vp["distance_along_shape"])

shape_geom = shapes.set_index("shape_id")["geometry"]
vp_shape_ids = vp["ROUTE_ID"].str.strip().map(SHAPE_KEY_TO_SHAPE_ID_MAP)
snap_distances = vp.groupby(vp_shape_ids, group_keys=False).apply(
    lambda g: g.geometry.map(
        lambda pt: pt.distance(shape_geom[g.name].interpolate(shape_geom[g.name].project(pt)))
    )
)
print("\nDistance from vehicle position to projected shape (meters) — all points:")
print(snap_distances.describe())
print()
print(f"Distance from vehicle position to projected shape (meters) — within {MAX_SNAP_DISTANCE_M}m threshold:")
print(snap_distances[vp["distance_along_shape"].notna()].describe())
print()

# Pick the first trip with enough valid pings to smooth (default k=20)
valid_counts = vp.groupby("TRIP_KEY")["distance_along_shape_smoothed"].count()
trip_key = valid_counts[valid_counts >= 20].index[0]
print(f"Plotting trip: {trip_key}")

trip = vp[vp["TRIP_KEY"] == trip_key].sort_values("event_time_datetime")

route_id = trip["ROUTE_ID"].iloc[0].strip()
trip_shape = shape_geom[[SHAPE_KEY_TO_SHAPE_ID_MAP[route_id]]]
signal_distances = project_signals_on_shape(signals, trip_shape, MAX_SNAP_DISTANCE_M)
nearside_stop_distances = project_signals_on_shape(stops[stops["nearside"] == True], trip_shape, MAX_SNAP_DISTANCE_M)
farside_stop_distances = project_signals_on_shape(stops[stops["nearside"] != True], trip_shape, MAX_SNAP_DISTANCE_M)

M_PER_MILE = 1609.344


fig, ax = plt.subplots(figsize=(4, 10))
ax.plot(trip["distance_along_shape"] / M_PER_MILE, trip["event_time_datetime"], marker="o", markersize=3, linestyle="None")
ax.plot(trip["distance_along_shape_smoothed"] / M_PER_MILE, trip["event_time_datetime"], color="red", linewidth=1.5)
for i, d in enumerate(signal_distances):
    ax.axvline(x=d / M_PER_MILE, color="gray", linewidth=0.5, linestyle="-", alpha=0.6, label="signal" if i == 0 else None)
for i, d in enumerate(nearside_stop_distances):
    ax.axvline(x=d / M_PER_MILE, color="blue", linewidth=0.5, linestyle="--", alpha=0.6, label="near-side stop" if i == 0 else None)
for i, d in enumerate(farside_stop_distances):
    ax.axvline(x=d / M_PER_MILE, color="pink", linewidth=0.5, linestyle="--", alpha=0.6, label="far-side stop" if i == 0 else None)
ax.set_xlabel("Distance along shape (mi)")
ax.set_ylabel("Time")
ax.set_title(f"Trip {trip_key} — smoothed distance along shape over time")
ax.legend()
plt.tight_layout()
plt.savefig("test_distance_along_shape_smoothed.png", dpi=150)
print("Saved test_distance_along_shape_smoothed.png")
plt.show()
