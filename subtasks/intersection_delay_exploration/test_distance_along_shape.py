import matplotlib.pyplot as plt

from constants import CULVER_CITY_FEED_KEY, MAX_SHAPE_JUMP_M, MAX_SNAP_DISTANCE_M, SERVICE_DATE, SHAPE_KEY_TO_SHAPE_ID_MAP
from _data_loaders import get_culver_city_vehicle_positions, get_selected_shapes
from match_shapes_vp import project_vp_on_shape
from smooth_trajectory import smooth_distances_per_trip

route_ids = list(SHAPE_KEY_TO_SHAPE_ID_MAP.keys())
shape_ids = list(SHAPE_KEY_TO_SHAPE_ID_MAP.values())

print("Loading vehicle positions...")
vp = get_culver_city_vehicle_positions(route_ids)

print("Loading shapes...")
shapes = get_selected_shapes(SERVICE_DATE, CULVER_CITY_FEED_KEY, shape_ids)

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

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trip["event_time_datetime"], trip["distance_along_shape"], marker="o", markersize=3)
ax.set_xlabel("Time")
ax.set_ylabel("Distance along shape (m)")
ax.set_title(f"Trip {trip_key} — distance along shape over time")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("test_distance_along_shape.png", dpi=150)
print("Saved test_distance_along_shape.png")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trip["event_time_datetime"], trip["distance_along_shape"], marker="o", markersize=3, linestyle="None", label="raw projection")
ax.plot(trip["event_time_datetime"], trip["distance_along_shape_smoothed"], color="red", linewidth=1.5, label="LOCREG-PCHIP smoothed")
ax.set_xlabel("Time")
ax.set_ylabel("Distance along shape (m)")
ax.set_title(f"Trip {trip_key} — smoothed distance along shape over time")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("test_distance_along_shape_smoothed.png", dpi=150)
print("Saved test_distance_along_shape_smoothed.png")
plt.show()
