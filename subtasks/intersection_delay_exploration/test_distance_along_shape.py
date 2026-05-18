import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import CA_NAD83_Albers, CULVER_CITY_FEED_KEY, MAX_SHAPE_JUMP_M, MAX_SNAP_DISTANCE_M, SERVICE_DATE, SHAPE_KEY_TO_SHAPE_ID_MAP
from _data_loaders import get_culver_city_vehicle_positions, get_selected_shapes, get_traffic_signals
from match_shapes_vp import project_vp_on_shape, project_points_on_shape
from smooth_trajectory import smooth_distances_per_trip, compute_speeds_per_trip
from delay_calculation import estimate_intersection_time_single_trip

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
vp_smoothed = smooth_distances_per_trip(vp, vp["distance_along_shape"], freq_seconds=1.0)

print("Computing speeds...")
vp_speeds = compute_speeds_per_trip(vp_smoothed, freq_seconds=1.0)

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

# Pick the first trip that produced smoothed samples
trip_key = vp_smoothed["TRIP_KEY"].iloc[0]
print(f"Plotting trip: {trip_key}")

trip = vp[vp["TRIP_KEY"] == trip_key].sort_values("event_time_datetime")
trip_smoothed = vp_smoothed[vp_smoothed["TRIP_KEY"] == trip_key].sort_values("event_time_datetime")
trip_speeds = vp_speeds[vp_speeds["TRIP_KEY"] == trip_key].sort_values("event_time_datetime")

route_id = trip["ROUTE_ID"].iloc[0].strip()
trip_shape = shape_geom[[SHAPE_KEY_TO_SHAPE_ID_MAP[route_id]]]
signal_distances = project_points_on_shape(signals, trip_shape, MAX_SNAP_DISTANCE_M)
nearside_stop_distances = project_points_on_shape(stops[stops["nearside"] == True], trip_shape, MAX_SNAP_DISTANCE_M)
farside_stop_distances = project_points_on_shape(stops[stops["nearside"] != True], trip_shape, MAX_SNAP_DISTANCE_M)

print("Estimating per-signal delay...")
NEARSIDE_STOP_DISTANCE_M = 30
SIGNAL_EFFECT_DISTANCE_M = 100
trip_stops_projected = project_points_on_shape(stops, trip_shape, MAX_SNAP_DISTANCE_M)
trip_stops_nearside = stops.loc[trip_stops_projected.index, "nearside"]
signal_delays_s = estimate_intersection_time_single_trip(
    smoothed_trajectory=trip_smoothed.set_index("event_time_datetime")["distance_along_shape_smoothed"],
    smoothed_speeds_mps=trip_speeds.set_index("event_time_datetime")["speed_m_per_s"],
    stops_projected=trip_stops_projected,
    stops_nearside=trip_stops_nearside,
    signals_projected=signal_distances,
    nearside_stop_distance=NEARSIDE_STOP_DISTANCE_M,
    signal_effect_distance=SIGNAL_EFFECT_DISTANCE_M,
)

M_PER_MILE = 1609.344
SECONDS_PER_HOUR = 3600
HIGHLIGHT_DELAY_THRESHOLD_S = 1.0


fig_dist, ax_dist = plt.subplots(figsize=(4, 10))
ax_dist.plot(trip["distance_along_shape"] / M_PER_MILE, trip["event_time_datetime"], marker="o", markersize=3, linestyle="None")
ax_dist.plot(trip_smoothed["distance_along_shape_smoothed"] / M_PER_MILE, trip_smoothed["event_time_datetime"], color="red", linewidth=1.5)
for i, (signal_id, d) in enumerate(signal_distances.items()):
    delay_s = signal_delays_s.loc[signal_id]
    is_high_delay_signal = delay_s > HIGHLIGHT_DELAY_THRESHOLD_S
    ax_dist.axvline(
        x=d / M_PER_MILE,
        color="black" if is_high_delay_signal else "gray",
        linewidth=1.5 if is_high_delay_signal else 0.5,
        linestyle="-",
        alpha=0.8 if is_high_delay_signal else 0.6,
        label="signal" if i == 0 else None,
    )
    ax_dist.text(
        d / M_PER_MILE, 0.98, f"{delay_s:.0f}s",
        transform=ax_dist.get_xaxis_transform(),
        rotation=90, va="top", ha="right",
        fontsize=8 if is_high_delay_signal else 7,
        color="black" if is_high_delay_signal else "gray",
        fontweight="bold" if is_high_delay_signal else "normal",
    )
for i, d in enumerate(nearside_stop_distances):
    ax_dist.axvline(x=d / M_PER_MILE, color="blue", linewidth=0.5, linestyle="--", alpha=0.6, label="near-side stop" if i == 0 else None)
for i, d in enumerate(farside_stop_distances):
    ax_dist.axvline(x=d / M_PER_MILE, color="pink", linewidth=0.5, linestyle="--", alpha=0.6, label="far-side stop" if i == 0 else None)
ax_dist.set_xlabel("Distance along shape (mi)")
ax_dist.set_ylabel("Time")
ax_dist.set_title(f"Trip {trip_key} — smoothed distance over time")
ax_dist.legend()
fig_dist.tight_layout()
fig_dist.savefig("test_distance_along_shape_smoothed.png", dpi=150)
print("Saved test_distance_along_shape_smoothed.png")

fig_speed, ax_speed = plt.subplots(figsize=(10, 4))
speed_mph = trip_speeds["speed_m_per_s"] * SECONDS_PER_HOUR / M_PER_MILE
ax_speed.plot(trip_speeds["event_time_datetime"], speed_mph, color="red", linewidth=1.5)

# Project each signal/stop distance to the time the bus crossed that distance
trip_distances_m_array = trip_smoothed["distance_along_shape_smoothed"].to_numpy()
trip_times_ns_array = trip_smoothed["event_time_datetime"].astype("int64").to_numpy()

def distances_to_crossing_times(distances_series: pd.Series) -> pd.Series:
    in_trip_range = (distances_series >= trip_distances_m_array[0]) & (distances_series <= trip_distances_m_array[-1])
    valid_distances = distances_series[in_trip_range]
    crossing_times_ns = np.interp(valid_distances.to_numpy(), trip_distances_m_array, trip_times_ns_array)
    return pd.Series(pd.to_datetime(crossing_times_ns.astype("int64"), unit="ns"), index=valid_distances.index)

signal_crossing_times = distances_to_crossing_times(signal_distances)
nearside_stop_crossing_times = distances_to_crossing_times(nearside_stop_distances)
farside_stop_crossing_times = distances_to_crossing_times(farside_stop_distances)

for i, (signal_id, crossing_time) in enumerate(signal_crossing_times.items()):
    delay_s = signal_delays_s.loc[signal_id]
    is_high_delay_signal = delay_s > HIGHLIGHT_DELAY_THRESHOLD_S
    ax_speed.axvline(
        x=crossing_time,
        color="black" if is_high_delay_signal else "gray",
        linewidth=1.5 if is_high_delay_signal else 0.5,
        linestyle="-",
        alpha=0.8 if is_high_delay_signal else 0.6,
        label="signal" if i == 0 else None,
    )
    ax_speed.text(
        crossing_time, 0.98, f"{delay_s:.0f}s",
        transform=ax_speed.get_xaxis_transform(),
        rotation=90, va="top", ha="right",
        fontsize=8 if is_high_delay_signal else 7,
        color="black" if is_high_delay_signal else "gray",
        fontweight="bold" if is_high_delay_signal else "normal",
    )
for i, crossing_time in enumerate(nearside_stop_crossing_times):
    ax_speed.axvline(x=crossing_time, color="blue", linewidth=0.5, linestyle="--", alpha=0.6, label="near-side stop" if i == 0 else None)
for i, crossing_time in enumerate(farside_stop_crossing_times):
    ax_speed.axvline(x=crossing_time, color="pink", linewidth=0.5, linestyle="--", alpha=0.6, label="far-side stop" if i == 0 else None)
ax_speed.set_xlabel("Time")
ax_speed.set_ylabel("Speed (mph)")
ax_speed.set_title(f"Trip {trip_key} — smoothed speed over time")
ax_speed.legend()
fig_speed.autofmt_xdate()
fig_speed.tight_layout()
fig_speed.savefig("test_speed_over_time_smoothed.png", dpi=150)
print("Saved test_speed_over_time_smoothed.png")

# Folium map: signals, stops, and speed-colored trip path
trip_shape_line = trip_shape.iloc[0]
trip_path_points_geodesic = gpd.GeoSeries(
    [trip_shape_line.interpolate(d) for d in trip_smoothed["distance_along_shape_smoothed"]],
    crs=CA_NAD83_Albers,
).to_crs("EPSG:4326")

speed_values_mph = speed_mph.to_numpy()
speed_colormap = plt.get_cmap("RdYlBu_r")  # blue=slow, red=fast
speed_color_norm = mcolors.Normalize(vmin=speed_values_mph.min(), vmax=speed_values_mph.max())

map_center_lat = trip_path_points_geodesic.geometry.y.mean()
map_center_lon = trip_path_points_geodesic.geometry.x.mean()
folium_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=14, tiles="CartoDB positron")

for i in range(len(trip_path_points_geodesic) - 1):
    start_point = trip_path_points_geodesic.iloc[i]
    end_point = trip_path_points_geodesic.iloc[i + 1]
    segment_speed_mph = (speed_values_mph[i] + speed_values_mph[i + 1]) / 2
    folium.PolyLine(
        locations=[(start_point.y, start_point.x), (end_point.y, end_point.x)],
        color=mcolors.to_hex(speed_colormap(speed_color_norm(segment_speed_mph))),
        weight=5,
        opacity=0.85,
        tooltip=f"{segment_speed_mph:.1f} mph",
    ).add_to(folium_map)

signals_near_shape_geodesic = (
    signals.loc[signal_distances.index]
    .to_crs("EPSG:4326")
    .assign(delay_text=signal_delays_s.apply(lambda s: f"{s:.1f} s"))
)
is_high_delay_signal = signal_delays_s > HIGHLIGHT_DELAY_THRESHOLD_S
low_delay_signals_geodesic = signals_near_shape_geodesic.loc[~is_high_delay_signal]
high_delay_signals_geodesic = signals_near_shape_geodesic.loc[is_high_delay_signal]
if not low_delay_signals_geodesic.empty:
    folium.GeoJson(
        low_delay_signals_geodesic,
        marker=folium.CircleMarker(radius=3, color="gray", fill=True, fill_opacity=0.85),
        tooltip=folium.GeoJsonTooltip(fields=["delay_text"], aliases=["signal delay"]),
    ).add_to(folium_map)
if not high_delay_signals_geodesic.empty:
    folium.GeoJson(
        high_delay_signals_geodesic,
        marker=folium.CircleMarker(radius=5, color="black", weight=2, fill=True, fill_color="gray", fill_opacity=0.85),
        tooltip=folium.GeoJsonTooltip(fields=["delay_text"], aliases=["signal delay"]),
    ).add_to(folium_map)

nearside_stops_geodesic = stops.loc[nearside_stop_distances.index].to_crs("EPSG:4326")
folium.GeoJson(
    nearside_stops_geodesic,
    marker=folium.CircleMarker(radius=4, color="blue", fill=True, fill_opacity=0.85),
    tooltip=folium.GeoJsonTooltip(fields=["stop_name"], aliases=["near-side stop"]),
).add_to(folium_map)

farside_stops_geodesic = stops.loc[farside_stop_distances.index].to_crs("EPSG:4326")
folium.GeoJson(
    farside_stops_geodesic,
    marker=folium.CircleMarker(radius=4, color="pink", fill=True, fill_opacity=0.85),
    tooltip=folium.GeoJsonTooltip(fields=["stop_name"], aliases=["far-side stop"]),
).add_to(folium_map)

folium_map.save("test_trip_speeds_map.html")
print("Saved test_trip_speeds_map.html")

plt.show()
