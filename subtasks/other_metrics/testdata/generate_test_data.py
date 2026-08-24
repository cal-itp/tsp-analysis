"""Generate synthetic mart_gtfs test data for the trip duration variability metric.

Produces one parquet file per warehouse table, column-for-column and type-for-type
identical to the real tables:

    mart_gtfs.fct_scheduled_trips
    mart_gtfs.fct_daily_scheduled_stops
    mart_gtfs.fct_stop_time_metrics

The scenario is a single service pattern: TRIP_COUNT trips departing every
HEADWAY_MINUTES minutes over a WINDOW_HOURS window, calling at N_STOPS stops.
There is no jitter anywhere. Every vehicle runs exactly to schedule, so the
observed headway at every stop is exactly HEADWAY_MINUTES, frequency is exactly
60 / HEADWAY_MINUTES vehicles per hour, and stop-to-stop travel time is
identical on every trip. That makes the expected value of a frequency or
headway calculation known up front.

Fields that link the tables to each other, or to the vp stop times tables, use
the real identifiers observed in trip_duration_variability.ipynb (feed_key,
shape_id, stop_id, base64_url). Everything else is arbitrary but type-correct.

Run with the project venv:

    uv run python testdata/generate_test_data.py
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from reference_tables import write_reference_tables
from schemas import SCHEMAS

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OUT_DIR = pathlib.Path(__file__).parent

SERVICE_DATE = datetime.date(2026, 6, 17)
# both feed keys appear in the notebook; d714... is the one in effect on 6/17
FEED_KEY = "d71492afe126cd61f629b17c3c056fcd"
PRIOR_FEED_KEY = "4dfefaf518766d1cfc3d254a65e4295e"  # in effect 6/15, unused here
# the notebook returns ECR2068 and ECR2069 (the two directions of the ECR route)
SHAPE_ID = "ECR2068"
ROUTE_ID = "ECR"
ROUTE_SHORT_NAME = "ECR"
GTFS_DATASET_NAME = "Bay Area 511 SamTrans Schedule"
GTFS_DATASET_KEY = "3f8ad2e07b8a4a1eb1e3b6a1c0d5f6a2"
SHAPE_ARRAY_KEY = "b2d6e9f10c3a4857be91d2f3a6c8047e"
FEED_TIMEZONE = "America/Los_Angeles"

# base64_url is the join key between the schedule and RT sides of the warehouse.
# the notebook's value is truncated in the cell output but its prefix matches the
# 511 datafeeds URL scheme, so reconstruct it the same way the pipeline does.
SCHEDULE_URL = "https://api.511.org/transit/datafeeds?operator_id=SM"
TRIP_UPDATES_URL = "https://api.511.org/transit/tripupdates?agency=SM"
SCHEDULE_BASE64_URL = base64.b64encode(SCHEDULE_URL.encode()).decode()
RT_BASE64_URL = base64.b64encode(TRIP_UPDATES_URL.encode()).decode()

FEED_VALID_FROM = datetime.datetime(2026, 6, 16, 3, 0, 0, 395507, tzinfo=datetime.UTC)

# 4 hours of service, a trip every 15 minutes, 10 stops per trip
WINDOW_HOURS = 4
HEADWAY_MINUTES = 15
TRIP_COUNT = WINDOW_HOURS * 60 // HEADWAY_MINUTES  # 16
N_STOPS = 10
FIRST_DEPARTURE = datetime.datetime(2026, 6, 17, 6, 0, 0)  # pacific, naive
PACIFIC_UTC_OFFSET = datetime.timedelta(hours=7)  # PDT in June

SCHEDULED_SEGMENT_SECONDS = 120  # stop-to-stop scheduled run time
SCHEDULED_DWELL_SECONDS = 30

# real ECR stop_ids from the notebook's schedule_rt_stop_times output
STOP_IDS = [
    "333505",
    "334087",
    "334410",
    "334077",
    "334090",
    "334079",
    "333511",
    "332214",
    "332171",
    "332019",
]
assert len(STOP_IDS) == N_STOPS

# the first, fourth and last stops are timepoints. dim_stop_times carries the
# flag; the notebook only treats a stop as a timepoint if it is one on every
# trip, so the flag is constant across trips.
TIMEPOINT_SEQUENCES = (1, 4, N_STOPS)

# the notebook's ECR2068 shape runs roughly between these two endpoints; the
# stops are spread evenly along that line so pt_geom is plausible
CORRIDOR_START = (-122.468786, 37.706203)
CORRIDOR_END = (-122.165986, 37.443948)

# length of the prediction window backing the fct_stop_time_metrics arrays
N_PREDICTION_MINUTES = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DBT_NULL = "_dbt_utils_surrogate_key_null_"


def surrogate_key(*fields) -> str:
    """Reimplementation of dbt_utils.generate_surrogate_key.

    md5 over the fields cast to string and joined by '-', with nulls replaced by
    a sentinel. Keeps the generated keys consistent with how the warehouse would
    have built them for this data.
    """
    parts = [DBT_NULL if f is None else str(f) for f in fields]
    return hashlib.md5("-".join(parts).encode()).hexdigest()


def time_of_day(hour: int) -> str:
    """Mirrors the generate_time_of_day_column macro."""
    if hour < 4:
        return "owl"
    if hour < 7:
        return "early_am"
    if hour < 10:
        return "am_peak"
    if hour < 15:
        return "midday"
    if hour < 20:
        return "pm_peak"
    return "evening"


# hours covered by each time of day bucket, per generate_time_of_day_hours
TIME_OF_DAY_HOURS = {
    "owl": 4,
    "early_am": 3,
    "am_peak": 3,
    "midday": 5,
    "pm_peak": 5,
    "evening": 4,
}


def stop_point_wkt(i: int) -> str:
    """WKT point for stop i, interpolated along the corridor."""
    t = i / (N_STOPS - 1)
    lon = CORRIDOR_START[0] + t * (CORRIDOR_END[0] - CORRIDOR_START[0])
    lat = CORRIDOR_START[1] + t * (CORRIDOR_END[1] - CORRIDOR_START[1])
    return f"POINT({lon:.6f} {lat:.6f})"


def gtfs_seconds(dt: datetime.datetime) -> int:
    """Seconds since noon-minus-twelve-hours on the service date."""
    midnight = datetime.datetime.combine(SERVICE_DATE, datetime.time())
    return int((dt - midnight).total_seconds())


def trip_id(i: int) -> str:
    """Mimics the SamTrans trip_id shape seen in the notebook."""
    return f"{15300000 + i}-154-Blocks-Weekday-60"


ARROW_TYPES = {
    "STRING": pa.string(),
    "INT64": pa.int64(),
    "FLOAT64": pa.float64(),
    "BOOL": pa.bool_(),
    "DATE": pa.date32(),
    "DATETIME": pa.timestamp("us"),
    "TIMESTAMP": pa.timestamp("us", tz="UTC"),
    "GEOGRAPHY": pa.string(),  # loaded into BigQuery as WKT
}


def arrow_type(bq_type: str) -> pa.DataType:
    if bq_type.startswith("ARRAY<"):
        return pa.list_(ARROW_TYPES[bq_type[len("ARRAY<") : -1]])
    return ARROW_TYPES[bq_type]


def bq_schema(table: str) -> list[dict]:
    """BigQuery load schema JSON, so `bq load` reproduces the exact column types."""
    fields = []
    for name, bq_type in SCHEMAS[table]:
        if bq_type.startswith("ARRAY<"):
            fields.append(
                {"name": name, "type": bq_type[len("ARRAY<") : -1], "mode": "REPEATED"}
            )
        else:
            fields.append({"name": name, "type": bq_type, "mode": "NULLABLE"})
    return fields


def write(table: str, rows: list[dict]) -> None:
    """Write rows to parquet in the warehouse's column order and types."""
    schema = pa.schema(
        [pa.field(name, arrow_type(bq_type)) for name, bq_type in SCHEMAS[table]]
    )
    missing = {name for name, _ in SCHEMAS[table]} - set(rows[0])
    extra = set(rows[0]) - {name for name, _ in SCHEMAS[table]}
    assert not missing, f"{table} missing columns: {sorted(missing)}"
    assert not extra, f"{table} has unknown columns: {sorted(extra)}"

    columns = {name: [row[name] for row in rows] for name, _ in SCHEMAS[table]}
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), OUT_DIR / f"{table}.parquet")
    (OUT_DIR / f"{table}.schema.json").write_text(json.dumps(bq_schema(table), indent=2) + "\n")
    print(f"{table}: {len(rows)} rows, {len(schema)} columns")


# ---------------------------------------------------------------------------
# The schedule: one trip every HEADWAY_MINUTES, N_STOPS stops each
# ---------------------------------------------------------------------------


def build_schedule() -> list[dict]:
    """Per trip and stop: scheduled arrival/departure, pacific naive datetimes."""
    schedule = []
    for t in range(TRIP_COUNT):
        start = FIRST_DEPARTURE + datetime.timedelta(minutes=HEADWAY_MINUTES * t)
        stop_times = []
        cursor = start
        for s in range(N_STOPS):
            arrival = cursor
            departure = arrival + datetime.timedelta(seconds=SCHEDULED_DWELL_SECONDS)
            stop_times.append(
                {
                    "stop_sequence": s + 1,
                    "stop_id": STOP_IDS[s],
                    "arrival": arrival,
                    "departure": departure,
                }
            )
            cursor = departure + datetime.timedelta(seconds=SCHEDULED_SEGMENT_SECONDS)
        schedule.append({"trip_index": t, "trip_id": trip_id(t), "stop_times": stop_times})
    return schedule


SCHEDULE = build_schedule()


# ---------------------------------------------------------------------------
# fct_scheduled_trips: one row per trip
# ---------------------------------------------------------------------------


def build_scheduled_trips() -> list[dict]:
    rows = []
    for trip in SCHEDULE:
        stop_times = trip["stop_times"]
        first_departure = stop_times[0]["departure"]
        last_arrival = stop_times[-1]["arrival"]
        first_departure_sec = gtfs_seconds(first_departure)
        last_arrival_sec = gtfs_seconds(last_arrival)
        first_departure_ts = first_departure + PACIFIC_UTC_OFFSET
        last_arrival_ts = last_arrival + PACIFIC_UTC_OFFSET

        # trip_key here is the schedule-side key (from dim_trips); it is
        # deliberately NOT the same value as fct_stop_time_metrics.trip_key,
        # which is built from the RT feed url. the tables join on
        # service_date + trip_id, as the notebook does.
        trip_key = surrogate_key(FEED_KEY, trip["trip_id"])

        rows.append(
            {
                "key": surrogate_key(SERVICE_DATE, FEED_KEY, trip["trip_id"]),
                "trip_instance_key": surrogate_key(
                    SERVICE_DATE, SCHEDULE_BASE64_URL, trip["trip_id"], 1
                ),
                "feed_timezone": FEED_TIMEZONE,
                "base64_url": SCHEDULE_BASE64_URL,
                "name": GTFS_DATASET_NAME,
                "regional_feed_type": "Combined Regional Feed",
                "gtfs_dataset_key": GTFS_DATASET_KEY,
                "service_date": SERVICE_DATE,
                "feed_key": FEED_KEY,
                "service_id": "Weekday-60",
                "trip_key": trip_key,
                "trip_id": trip["trip_id"],
                "iteration_num": 1,
                "frequencies_defined_trip": False,
                "trip_short_name": None,
                "direction_id": 0,
                "block_id": f"154-{trip['trip_index'] % 4 + 1}",
                "route_key": "9a1c8d3f4b7e2a6c5d0f1b8e3a4c7d29",
                "route_id": ROUTE_ID,
                "route_type": "3",
                "route_short_name": ROUTE_SHORT_NAME,
                "route_long_name": "El Camino Real",
                "route_continuous_pickup": 1,
                "route_continuous_drop_off": 1,
                "route_desc": None,
                "route_color": "005C9E",
                "route_text_color": "FFFFFF",
                "agency_id": "SM",
                "network_id": None,
                "shape_array_key": SHAPE_ARRAY_KEY,
                "shape_id": SHAPE_ID,
                "contains_warning_duplicate_trip_primary_key": False,
                "num_distinct_stops_served": N_STOPS,
                "num_stop_times": N_STOPS,
                "trip_first_departure_sec": first_departure_sec,
                "trip_last_arrival_sec": last_arrival_sec,
                "trip_start_timezone": FEED_TIMEZONE,
                "trip_end_timezone": FEED_TIMEZONE,
                "service_hours": round((last_arrival_sec - first_departure_sec) / 3600, 4),
                "flex_service_hours": 0.0,
                "contains_warning_duplicate_stop_times_primary_key": False,
                "contains_warning_missing_foreign_key_stop_id": False,
                "trip_first_departure_ts": first_departure_ts.replace(tzinfo=datetime.UTC),
                "trip_last_arrival_ts": last_arrival_ts.replace(tzinfo=datetime.UTC),
                "first_start_pickup_drop_off_window_sec": None,
                "last_end_pickup_drop_off_window_sec": None,
                "is_gtfs_flex_trip": False,
                "is_entirely_demand_responsive_trip": False,
                "num_gtfs_flex_stop_times": 0,
                "num_approximate_timepoint_stop_times": N_STOPS - len(TIMEPOINT_SEQUENCES),
                "num_exact_timepoint_stop_times": len(TIMEPOINT_SEQUENCES),
                "num_arrival_times_populated_stop_times": N_STOPS,
                "num_departure_times_populated_stop_times": N_STOPS,
                "trip_first_start_pickup_drop_off_window_ts": None,
                "trip_last_end_pickup_drop_off_window_ts": None,
                "trip_start_date_pacific": SERVICE_DATE,
                "trip_first_departure_datetime_pacific": first_departure,
                "trip_last_arrival_datetime_pacific": last_arrival,
                "trip_start_date_local_tz": SERVICE_DATE,
                "trip_first_departure_datetime_local_tz": first_departure,
                "trip_last_arrival_datetime_local_tz": last_arrival,
                "trip_first_start_pickup_drop_off_window_date_pacific": None,
                "trip_first_start_pickup_drop_off_window_datetime_pacific": None,
                "trip_last_end_pickup_drop_off_window_pacific": None,
                "trip_first_start_pickup_drop_off_window_date_local_tz": None,
                "trip_first_start_pickup_drop_off_window_datetime_local_tz": None,
                "trip_last_end_pickup_drop_off_window_datetime_local_tz": None,
                "time_of_day": time_of_day(first_departure.hour),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# fct_daily_scheduled_stops: one row per stop per service date
# ---------------------------------------------------------------------------


def build_daily_scheduled_stops() -> list[dict]:
    rows = []
    for s in range(N_STOPS):
        arrivals = [trip["stop_times"][s]["arrival"] for trip in SCHEDULE]
        departures = [trip["stop_times"][s]["departure"] for trip in SCHEDULE]

        by_bucket: dict[str, int] = {bucket: 0 for bucket in TIME_OF_DAY_HOURS}
        for arrival in arrivals:
            by_bucket[time_of_day(arrival.hour)] += 1
        per_hour = {
            bucket: round(count / TIME_OF_DAY_HOURS[bucket], 1)
            for bucket, count in by_bucket.items()
        }

        stop_id = STOP_IDS[s]
        rows.append(
            {
                "key": surrogate_key(SERVICE_DATE, surrogate_key(FEED_KEY, stop_id)),
                "service_date": SERVICE_DATE,
                "feed_key": FEED_KEY,
                "stop_id": stop_id,
                "feed_timezone": FEED_TIMEZONE,
                "daily_arrivals": len(arrivals),
                "first_stop_arrival_datetime_pacific": min(arrivals),
                "last_stop_departure_datetime_pacific": max(departures),
                "_feed_valid_from": FEED_VALID_FROM,
                "n_hours_in_service": len({arrival.hour for arrival in arrivals}),
                "arrivals_per_hour_owl": per_hour["owl"],
                "arrivals_per_hour_early_am": per_hour["early_am"],
                "arrivals_per_hour_am_peak": per_hour["am_peak"],
                "arrivals_per_hour_midday": per_hour["midday"],
                "arrivals_per_hour_pm_peak": per_hour["pm_peak"],
                "arrivals_per_hour_evening": per_hour["evening"],
                "arrivals_owl": by_bucket["owl"],
                "arrivals_early_am": by_bucket["early_am"],
                "arrivals_am_peak": by_bucket["am_peak"],
                "arrivals_midday": by_bucket["midday"],
                "arrivals_pm_peak": by_bucket["pm_peak"],
                "arrivals_evening": by_bucket["evening"],
                "route_type_0": 0,
                "route_type_1": 0,
                "route_type_2": 0,
                "route_type_3": len(arrivals),  # bus
                "route_type_4": 0,
                "route_type_5": 0,
                "route_type_6": 0,
                "route_type_7": 0,
                "route_type_11": 0,
                "route_type_12": 0,
                "missing_route_type": 0,
                "contains_warning_duplicate_stop_times_primary_key": False,
                "contains_warning_duplicate_trip_primary_key": False,
                "route_id_array": [ROUTE_ID],
                "route_type_array": [3],
                "n_route_types": 1,
                "contains_warning_duplicate_stop_primary_key": False,
                "stop_key": surrogate_key(FEED_KEY, stop_id),
                "tts_stop_name": None,
                "pt_geom": stop_point_wkt(s),
                "parent_station": None,
                "stop_code": stop_id,
                "stop_name": f"El Camino Real & Test St {s + 1}",
                "stop_desc": None,
                "location_type": 0,
                "stop_timezone_coalesced": FEED_TIMEZONE,
                "wheelchair_boarding": 1,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# fct_stop_time_metrics: one row per trip and stop, with RT actuals
# ---------------------------------------------------------------------------


def build_stop_time_metrics() -> list[dict]:
    rows = []
    for trip in SCHEDULE:
        stop_times = trip["stop_times"]

        trip_start_time = stop_times[0]["departure"].strftime("%H:%M:%S")
        trip_key = surrogate_key(
            SERVICE_DATE,
            RT_BASE64_URL,
            SCHEDULE_BASE64_URL,
            trip["trip_id"],
            trip_start_time,
        )

        for stop_time in stop_times:
            # every vehicle runs exactly to schedule, so the observed headway at
            # each stop is exactly HEADWAY_MINUTES and travel time between any
            # two stops is identical on every trip
            arrival = stop_time["arrival"]
            departure = stop_time["departure"]

            minutes_until_arrival = [float(m) for m in range(N_PREDICTION_MINUTES, 0, -1)]
            prediction_error = [0.0] * N_PREDICTION_MINUTES
            scaled_prediction_error = [0.0] * N_PREDICTION_MINUTES

            rows.append(
                {
                    "trip_key": trip_key,
                    "key": surrogate_key(
                        SERVICE_DATE,
                        RT_BASE64_URL,
                        trip["trip_id"],
                        stop_time["stop_id"],
                        stop_time["stop_sequence"],
                    ),
                    "service_date": SERVICE_DATE,
                    "base64_url": RT_BASE64_URL,
                    "schedule_base64_url": SCHEDULE_BASE64_URL,
                    "trip_id": trip["trip_id"],
                    "trip_start_time": trip_start_time,
                    "stop_id": stop_time["stop_id"],
                    "stop_sequence": stop_time["stop_sequence"],
                    "actual_arrival": arrival + PACIFIC_UTC_OFFSET,
                    "actual_departure": departure + PACIFIC_UTC_OFFSET,
                    "actual_arrival_pacific": arrival,
                    "actual_departure_pacific": departure,
                    # the prediction quality columns play no part in frequency or
                    # headway, so they are fixed to a perfectly accurate feed
                    "avg_prediction_error_sec": 0.0,
                    "avg_scaled_prediction_error_sec": 0.0,
                    "n_tu_accurate_minutes": N_PREDICTION_MINUTES,
                    "n_predictions_early": 0,
                    "n_predictions_ontime": N_PREDICTION_MINUTES,
                    "n_predictions_late": 0,
                    "prediction_error_by_minute_array": prediction_error,
                    "minutes_until_arrival_array": minutes_until_arrival,
                    "scaled_prediction_error_by_minute_array": scaled_prediction_error,
                    "n_tu_complete_minutes": N_PREDICTION_MINUTES,
                    "n_tu_minutes_available": N_PREDICTION_MINUTES,
                    "sum_prediction_spread_seconds": 0,
                    "max_minutes_until_arrival": float(N_PREDICTION_MINUTES),
                    "n_predictions": N_PREDICTION_MINUTES * 2,
                }
            )
    return rows


if __name__ == "__main__":
    write("fct_scheduled_trips", build_scheduled_trips())
    write("fct_daily_scheduled_stops", build_daily_scheduled_stops())
    write("fct_stop_time_metrics", build_stop_time_metrics())
    write_reference_tables(
        OUT_DIR,
        SCHEDULE,
        timepoint_sequences=TIMEPOINT_SEQUENCES,
        service_date=SERVICE_DATE,
        feed_key=FEED_KEY,
        shape_id=SHAPE_ID,
        stop_ids=STOP_IDS,
        schedule_base64_url=SCHEDULE_BASE64_URL,
        feed_valid_from=FEED_VALID_FROM,
        feed_timezone=FEED_TIMEZONE,
        gtfs_dataset_name=GTFS_DATASET_NAME,
        gtfs_dataset_key=GTFS_DATASET_KEY,
        shape_array_key=SHAPE_ARRAY_KEY,
        corridor=(CORRIDOR_START, CORRIDOR_END),
    )
