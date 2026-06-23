# `0_map_match_and_smooth.py` output schema

`0_map_match_and_smooth.py` reads the consolidated raw positions written by
`0_preprocess_vehicle_positions.py` and writes two parquets to GCS. Paths are in
`src/constants.py`; read them with the loaders in `src/_data_loaders.py`.

Both files cover **all trips on all service dates**. `TRIP_KEY` is only unique
within a service date, so a trip is identified by `(service_date, TRIP_KEY)` (or
equivalently `(event_time_datetime.dt.date, TRIP_KEY)` in the matched file).

---

## 1. Matched pings — `CULVER_CITY_MATCHED_VEHICLE_POSITIONS_PATH`

`culver_city_matched_vehicle_positions.parquet` — a **geoparquet** (carries
geometry), one row per raw vehicle-position ping that matched a known route.
Built by `build_matched_vehicle_positions`. Load with
`get_matched_vehicle_positions(service_date=None, shape_id=None)`.

It contains **every column from the consolidated raw positions**
(`CULVER_CITY_VEHICLE_POSITIONS_PATH`) plus the two columns the map-matching step
adds. Rows whose `ROUTE_ID` has no shape in `SHAPE_KEY_TO_SHAPE_ID_MAP` are
dropped.

Key columns:

| column | dtype | source | meaning |
|---|---|---|---|
| `shape_id` | string | **added** | GTFS shape the ping was matched to (e.g. `shp-1-05`), mapped from `ROUTE_ID` via `SHAPE_KEY_TO_SHAPE_ID_MAP` |
| `distance_along_shape` | float64 | **added** | meters along `shape_id` from its start; **NaN** where the ping is beyond `MAX_SNAP_DISTANCE_M` (20 m) of the shape or jumps more than `MAX_SHAPE_JUMP_M` (200 m) from the previous ping in the trip |
| `geometry` | point | raw | ping location, CRS `CA_NAD83_Albers` (EPSG:3310) |
| `event_time_datetime` | datetime64[ns] | raw | parsed `EVENT_TIME`; the service date is `event_time_datetime.dt.date` |
| `TRIP_KEY` | string | raw | trip id (unique within a service date) |
| `ROUTE_ID` | string | raw | VP route id (e.g. `105`, `104`); whitespace not stripped |

…plus all other raw VehicleState columns (`DWELL_TIME`, `STOP_BACK_DOOR_ENTRY`,
`STOP_SEQUENCE`, `LONGITUDE`/`LATITUDE`, etc.). Object columns are stored as
nullable `string`.

Notes:
- One row per raw ping (not resampled). Rows with `distance_along_shape == NaN`
  are kept — they are real pings that fell outside the snap/jump thresholds.
- `get_matched_vehicle_positions` returns the rows sorted by
  `["TRIP_KEY", "event_time_datetime"]`.

---

## 2. Smoothed trips — `CULVER_CITY_SMOOTHED_VEHICLE_POSITIONS_PATH`

`culver_city_smoothed_vehicle_positions.parquet` — a **plain (non-geo)**
DataFrame, one row per trip per smoothed time sample. Built by
`build_smoothed_vehicle_positions`. Load with
`get_smoothed_vehicle_positions(service_date=None, shape_id=None)`.

| column | dtype | meaning |
|---|---|---|
| `service_date` | string `"YYYY-MM-DD"` | the date the trip ran; with `TRIP_KEY`, uniquely identifies a trip |
| `shape_id` | string | GTFS shape the trip was matched to (e.g. `shp-1-05`) |
| `TRIP_KEY` | string | trip id (unique within `service_date`) |
| `event_time_datetime` | datetime64[ns] | sample timestamp, evenly spaced `SMOOTH_FREQ_SECONDS` (1.0 s) apart within each trip |
| `distance_along_shape_smoothed` | float64 | LOCREG-PCHIP smoothed distance along the shape, meters (filtered to ≥ 0) |
| `speed_m_per_s` | float64 | `np.gradient` of the smoothed distance, meters/second |

Notes:
- One row per second per trip on an even grid, starting at the trip's first
  valid ping and not exceeding the last — **not** aligned 1:1 with the raw pings
  in the matched file.
- `distance_along_shape_smoothed` and `speed_m_per_s` are aligned 1:1 (inner
  merge on `["TRIP_KEY", "event_time_datetime"]` within each date).
- Trips with fewer than `k = 20` valid pings are dropped by the smoother.
