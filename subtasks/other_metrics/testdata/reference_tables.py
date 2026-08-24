"""The dim and feed tables the notebook's templates join against.

generate_test_data.py and its siblings build the three fact tables that carry
the scenario. This module builds the four supporting tables that the templates
in ../templates/ also reference, so every table named in a query exists:

    mart_gtfs.dim_stop_times           timepoint flags, per trip and stop
    mart_gtfs.dim_schedule_feeds       feed_key -> base64_url and _valid_from
    mart_gtfs.fct_daily_schedule_feeds which feed was in effect on which date
    mart_gtfs.fct_daily_scheduled_shapes  the shape geometry

Each is column-for-column and type-for-type identical to the real table. Only
the columns the templates actually read carry meaning; the rest are arbitrary
but type-correct.

Two BigQuery types have no faithful parquet equivalent and are written as
strings, the same compromise already made for GEOGRAPHY:

    INTERVAL  written as a BigQuery interval literal, '0-0 0 6:2:0'
    NUMERIC   left null (no template reads one)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from schemas import SCHEMAS

DBT_NULL = "_dbt_utils_surrogate_key_null_"

ARROW_TYPES = {
    "STRING": pa.string(),
    "INT64": pa.int64(),
    "FLOAT64": pa.float64(),
    "BOOL": pa.bool_(),
    "DATE": pa.date32(),
    "DATETIME": pa.timestamp("us"),
    "TIMESTAMP": pa.timestamp("us", tz="UTC"),
    "GEOGRAPHY": pa.string(),  # WKT
    "INTERVAL": pa.string(),  # BigQuery interval literal
    "NUMERIC": pa.decimal128(38, 9),
}


def arrow_type(bq_type: str) -> pa.DataType:
    if bq_type.startswith("ARRAY<"):
        return pa.list_(ARROW_TYPES[bq_type[len("ARRAY<") : -1]])
    return ARROW_TYPES[bq_type]


def bq_schema(table: str) -> list[dict]:
    fields = []
    for name, bq_type in SCHEMAS[table]:
        if bq_type.startswith("ARRAY<"):
            fields.append(
                {"name": name, "type": bq_type[len("ARRAY<") : -1], "mode": "REPEATED"}
            )
        else:
            fields.append({"name": name, "type": bq_type, "mode": "NULLABLE"})
    return fields


def surrogate_key(*fields) -> str:
    parts = [DBT_NULL if f is None else str(f) for f in fields]
    return hashlib.md5("-".join(parts).encode()).hexdigest()


def write(out_dir: pathlib.Path, table: str, rows: list[dict]) -> None:
    schema = pa.schema(
        [pa.field(name, arrow_type(bq_type)) for name, bq_type in SCHEMAS[table]]
    )
    missing = {name for name, _ in SCHEMAS[table]} - set(rows[0])
    extra = set(rows[0]) - {name for name, _ in SCHEMAS[table]}
    assert not missing, f"{table} missing columns: {sorted(missing)}"
    assert not extra, f"{table} has unknown columns: {sorted(extra)}"

    out_dir.mkdir(exist_ok=True)
    columns = {name: [row[name] for row in rows] for name, _ in SCHEMAS[table]}
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), out_dir / f"{table}.parquet")
    (out_dir / f"{table}.schema.json").write_text(json.dumps(bq_schema(table), indent=2) + "\n")
    print(f"{table}: {len(rows)} rows, {len(schema)} columns")


def interval_literal(dt: datetime.datetime, service_date: datetime.date) -> str:
    """BigQuery INTERVAL literal for a stop time, as seconds since midnight."""
    total = int((dt - datetime.datetime.combine(service_date, datetime.time())).total_seconds())
    return f"0-0 0 {total // 3600}:{total % 3600 // 60}:{total % 60}"


def gtfs_time(dt: datetime.datetime, service_date: datetime.date) -> str:
    """GTFS HH:MM:SS, which may exceed 24 hours for trips that wrap midnight."""
    total = int((dt - datetime.datetime.combine(service_date, datetime.time())).total_seconds())
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def write_reference_tables(
    out_dir: pathlib.Path,
    schedule: list[dict],
    *,
    timepoint_sequences: tuple[int, ...],
    service_date: datetime.date,
    feed_key: str,
    shape_id: str,
    stop_ids: list[str],
    schedule_base64_url: str,
    feed_valid_from: datetime.datetime,
    feed_timezone: str,
    gtfs_dataset_name: str,
    gtfs_dataset_key: str,
    shape_array_key: str,
    corridor: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Build and write the four supporting tables for one scenario."""

    # --- dim_stop_times: one row per trip and stop -------------------------
    # the notebook derives timepoint status from this table, and only counts a
    # stop as a timepoint if it is one on every trip, so the flag is constant
    # across trips for a given stop_sequence
    stop_times_rows = []
    for trip in schedule:
        for stop_time in trip["stop_times"]:
            seq = stop_time["stop_sequence"]
            arrival, departure = stop_time["arrival"], stop_time["departure"]
            stop_times_rows.append(
                {
                    "key": surrogate_key(feed_key, trip["trip_id"], seq),
                    "_gtfs_key": surrogate_key(feed_key, trip["trip_id"], seq, "gtfs"),
                    "base64_url": schedule_base64_url,
                    "feed_key": feed_key,
                    "trip_id": trip["trip_id"],
                    "stop_id": stop_time["stop_id"],
                    "stop_sequence": seq,
                    "arrival_time": gtfs_time(arrival, service_date),
                    "departure_time": gtfs_time(departure, service_date),
                    "arrival_time_interval": interval_literal(arrival, service_date),
                    "departure_time_interval": interval_literal(departure, service_date),
                    "stop_headsign": None,
                    "pickup_type": 0,
                    "drop_off_type": 0,
                    "continuous_pickup": 1,
                    "continuous_drop_off": 1,
                    "shape_dist_traveled": None,
                    "timepoint": 1 if seq in timepoint_sequences else 0,
                    "warning_duplicate_gtfs_key": False,
                    "warning_missing_foreign_key_stop_id": False,
                    "_dt": service_date,
                    "_feed_valid_from": feed_valid_from,
                    "_line_number": seq,
                    "feed_timezone": feed_timezone,
                    "arrival_sec": int(
                        (arrival - datetime.datetime.combine(service_date, datetime.time())).total_seconds()
                    ),
                    "departure_sec": int(
                        (departure - datetime.datetime.combine(service_date, datetime.time())).total_seconds()
                    ),
                    "start_pickup_drop_off_window": None,
                    "end_pickup_drop_off_window": None,
                    "start_pickup_drop_off_window_interval": None,
                    "end_pickup_drop_off_window_interval": None,
                    "start_pickup_drop_off_window_sec": None,
                    "end_pickup_drop_off_window_sec": None,
                    "mean_duration_factor": None,
                    "mean_duration_offset": None,
                    "safe_duration_factor": None,
                    "safe_duration_offset": None,
                    "pickup_booking_rule_id": None,
                    "drop_off_booking_rule_id": None,
                }
            )
    write(out_dir, "dim_stop_times", stop_times_rows)

    # --- dim_schedule_feeds: one row per feed version ----------------------
    write(
        out_dir,
        "dim_schedule_feeds",
        [
            {
                "key": feed_key,
                "base64_url": schedule_base64_url,
                "download_success": True,
                "unzip_success": True,
                "zipfile_extract_md5hash": surrogate_key(feed_key, "zip"),
                "feed_timezone": feed_timezone,
                "_dt": service_date,
                "_valid_from": feed_valid_from,
                "_valid_to": datetime.datetime(2099, 1, 1, tzinfo=datetime.UTC),
                "_is_current": True,
            }
        ],
    )

    # --- fct_daily_schedule_feeds: one row per feed per service date -------
    write(
        out_dir,
        "fct_daily_schedule_feeds",
        [
            {
                "key": surrogate_key(service_date, feed_key),
                "date": datetime.datetime.combine(service_date, datetime.time()),
                "feed_key": feed_key,
                "feed_timezone": feed_timezone,
                "base64_url": schedule_base64_url,
                "_feed_valid_from": feed_valid_from,
                "gtfs_dataset_key": gtfs_dataset_key,
                "gtfs_dataset_name": gtfs_dataset_name,
            }
        ],
    )

    # --- fct_daily_scheduled_shapes: one row per shape per service date ----
    # 50 points along the corridor, so the notebook's make_linestring has
    # something to build a real LineString from
    start, end = corridor
    pt_array = []
    for i in range(50):
        t = i / 49
        lon = start[0] + t * (end[0] - start[0])
        lat = start[1] + t * (end[1] - start[1])
        pt_array.append(f"POINT({lon:.6f} {lat:.6f})")

    all_stop_times = [st for trip in schedule for st in trip["stop_times"]]
    write(
        out_dir,
        "fct_daily_scheduled_shapes",
        [
            {
                "key": surrogate_key(service_date, feed_key, shape_id),
                "feed_key": feed_key,
                "service_date": service_date,
                "shape_id": shape_id,
                "shape_array_key": shape_array_key,
                "feed_timezone": feed_timezone,
                "n_trips": len(schedule),
                "shape_first_departure_datetime_pacific": min(
                    st["departure"] for st in all_stop_times
                ),
                "shape_last_arrival_datetime_pacific": max(
                    st["arrival"] for st in all_stop_times
                ),
                "contains_warning_duplicate_trip_primary_key": False,
                "pt_array": pt_array,
            }
        ],
    )
