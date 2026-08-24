# Synthetic mart_gtfs test data

Fixture data shaped exactly like the warehouse tables the notebook reads. Three
datasets, identical in format — each one a complete set of every table the
templates in `../templates/` reference:

| Dataset | Location | What it pins down |
| --- | --- | --- |
| constant headway | this directory | a trip every 15 minutes for 4 hours |
| [changing headway](#second-dataset-changing-headway) | `variable_headway/` | every 10 minutes, then every 20, all day |
| [changing travel time](#third-dataset-changing-travel-time) | `variable_travel_time/` | a p80/p20 variability ratio of exactly 2.0 |

Each dataset holds seven tables. `<table>.schema.json` is the matching
`bq load` schema.

| File | Mirrors | Read by |
| --- | --- | --- |
| `fct_scheduled_trips.parquet` | `mart_gtfs.fct_scheduled_trips` | `shapes_routes.sql`, `schedule_rt_stop_times.sql` |
| `fct_daily_scheduled_stops.parquet` | `mart_gtfs.fct_daily_scheduled_stops` | `schedule_rt_stop_times.sql` |
| `fct_stop_time_metrics.parquet` | `mart_gtfs.fct_stop_time_metrics` | `schedule_rt_stop_times.sql` |
| `dim_stop_times.parquet` | `mart_gtfs.dim_stop_times` | `schedule_rt_stop_times.sql` |
| `dim_schedule_feeds.parquet` | `mart_gtfs.dim_schedule_feeds` | `schedule_feed_entry.sql` |
| `fct_daily_schedule_feeds.parquet` | `mart_gtfs.fct_daily_schedule_feeds` | `shapes_routes.sql` |
| `fct_daily_scheduled_shapes.parquet` | `mart_gtfs.fct_daily_scheduled_shapes` | `shape_geometry.sql` |

Every table has the full column list, in warehouse order, with matching
BigQuery types.

## Timepoints

`dim_stop_times.timepoint` is 1 at **stop_sequence 1, 4 and 10** — first,
fourth and last — and 0 elsewhere, in all three datasets. The flag is constant
across trips, because the notebook only accepts a stop as a timepoint when
`n_timepoint = n_stop_times`, i.e. when it is flagged on every trip.

That leaves the notebook measuring two segments, 1→4 and 4→10.
`fct_scheduled_trips.num_exact_timepoint_stop_times` is kept consistent at 3.

## Scenario

One service pattern on `2026-06-17`: a trip every 15 minutes for 4 hours
(06:00–09:45 departures, 16 trips), calling at 10 stops. Run time is a flat
120 s per segment with a 30 s dwell.

**There is no jitter.** Every vehicle runs exactly to schedule, so the answers a
frequency or headway calculation should produce are known up front:

| quantity | expected value |
| --- | --- |
| observed headway at every stop | exactly 15:00 |
| frequency at every stop | exactly 4 vehicles/hour |
| arrivals per stop over the window | 16 |
| stop-to-stop travel time | exactly 120 s on every trip |
| timepoint-to-timepoint time | exactly 420 s (1→4), 870 s (4→10) |
| p80/p20 travel time variability | exactly 1.0 |

The prediction-quality columns in `fct_stop_time_metrics`
(`avg_prediction_error_sec`, the `*_array` columns, the `n_predictions_*`
counts) are fixed to a perfectly accurate feed — they carry no signal and take
no part in frequency or headway.

## Identifiers

Fields that link the tables to each other, or to the vp stop times tables, use
real values pulled from `trip_duration_variability.ipynb`:

- `feed_key` `d71492afe126cd61f629b17c3c056fcd` (the SamTrans feed in effect on 6/17)
- `shape_id` `ECR2068`, `route_id`/`route_short_name` `ECR`
- the 10 `stop_id`s from the notebook's `schedule_rt_stop_times` output
- `schedule_base64_url`, rebuilt as base64 of the 511 datafeeds URL — the
  notebook's value is truncated in the cell output, but its prefix matches

Everything else (`key`, `route_key`, `shape_array_key`, `block_id`, stop names,
prediction-quality columns) is arbitrary. Surrogate keys are generated with a
Python reimplementation of `dbt_utils.generate_surrogate_key`, so they look and
behave like real ones. `fct_stop_time_metrics.trip_key` deliberately differs
from `fct_scheduled_trips.trip_key` — as in the warehouse, the two are built
from different inputs, and the tables join on `service_date` + `trip_id`.

Each dataset uses its own `trip_id` base — `153000xx`, `154000xx`, `155000xx` —
so no two fixtures ever share one.

## Second dataset: changing headway

`variable_headway/` runs all day at two different headways:

- **06:00 – 13:50**, a trip every 10 minutes — 48 trips
- **14:00 – 21:40**, a trip every 20 minutes — 24 trips

72 trips over the same 10 stops, 720 stop times. The block boundary carries the
outgoing headway: the last 10-minute trip departs 13:50 and the first 20-minute
trip departs 14:00. Still no jitter, so the expected values are again exact:

| quantity | expected value |
| --- | --- |
| headway gaps at every stop | 48 of exactly 10:00, then 23 of exactly 20:00 |
| frequency at every stop | 6 veh/hour for hours 06–13, 3 veh/hour for hours 14–21 |
| arrivals per stop over the day | 72 |
| stop-to-stop travel time | exactly 120 s on every trip |

Service spans hours 6–22, which exercises every `time_of_day` bucket in
`fct_daily_scheduled_stops` — early_am 6, am_peak 18, midday 27, pm_peak 15,
evening 6.

## Third dataset: changing travel time

`variable_travel_time/` is the 16-trip, 15-minute scenario again, but every
third trip runs the whole line in **half** the time — 60 s per segment and a
15 s dwell, against 120 s and 30 s on the rest. Travel time takes exactly two
values, so p80/p20 is exactly **2.0**:

| measured over | p80 | p20 | ratio |
| --- | --- | --- | --- |
| stop to stop, any segment | 120 s | 60 s | 2.0 |
| timepoint 1 → 4 | 420 s | 210 s | 2.0 |
| timepoint 4 → 10 | 870 s | 435 s | 2.0 |
| end to end trip duration | 1320 s | 660 s | 2.0 |

### Why one third and not 20%

A 20/80 split does **not** give a ratio of 2 — it gives 1.11. For `p80/p20` to
equal `slow/fast`, the 20th percentile has to land inside the fast group and the
80th inside the slow one. Under pandas' linear interpolation over `n` trips that
needs

```
0.2 + 0.8/n  <=  f  <=  0.8 - 0.8/n
```

and `f = 0.2` fails the lower bound at every `n`. Worse, no choice of fast
travel time rescues it: with only 20% fast, p20 stays inside the slow group, so
the ratio is capped at `1/0.8 = 1.25` even as the fast time approaches zero.

```
 n trips     f=0.20     f=0.25     f=0.33     f=0.50
      16      1.000      2.000      2.000      2.000
      20      1.111      2.000      2.000      2.000
      72      1.000      2.000      2.000      2.000
    1000      1.111      2.000      2.000      2.000
```

One third clears the bound at every trip count in use here, and still leaves
most trips on the even time. `FAST_TRIP_EVERY` at the top of the generator
controls it.

Note that fast trips run down the line quicker than slow ones, so past the first
stop the headway in this fixture is no longer uniform. Use it for travel time
variability, not for frequency or headway.

## Running the notebook against these

Every table the templates name now exists, so the queries have all their inputs.
Point the notebook at whichever dataset you loaded and narrow the parameters to
the fixture's single service date:

```python
TARGET_DATES = ["2026-06-17"]
TARGET_ROUTE_SHORT_NAME = "ECR"
TARGET_SCHEDULE_FEED = "Bay Area 511 SamTrans Schedule"
```

## Regenerating

```bash
cd testdata
uv run python generate_test_data.py                  # this directory
uv run python generate_variable_headway_data.py      # variable_headway/
uv run python generate_variable_travel_time_data.py  # variable_travel_time/
```

Constants at the top of each generator control the date, feed, shape, stop list,
timepoints, headway and window length — `HEADWAY_MINUTES` in the first,
`HEADWAY_BLOCKS` (a list of `(headway minutes, hours)`) in the second,
`FAST_TRIP_EVERY` and `FAST_SEGMENT_FACTOR` in the third. All are fully
deterministic; there is no RNG.

Two shared modules:

- `schemas.py` — column lists and types for all seven tables, extracted from the
  published cal-itp dbt catalog
  (<https://dbt-docs.dds.dot.ca.gov/catalog.json>, generated 2026-08-19)
- `reference_tables.py` — builds the four supporting tables, which differ
  between datasets only in their trip list

## Loading into BigQuery

```bash
bq load --source_format=PARQUET --replace \
  cal-itp-data-infra-staging:<your_dataset>.fct_stop_time_metrics \
  fct_stop_time_metrics.parquet
```

Parquet carries the types itself, so no schema flag is needed. Three caveats if
you want a byte-identical copy of the warehouse table:

- `pt_geom` and `pt_array` are written as WKT `STRING`. To land them as
  `GEOGRAPHY`, load into a staging table and
  `CREATE TABLE … AS SELECT * REPLACE(ST_GEOGFROMTEXT(pt_geom) AS pt_geom)`.
- `DATETIME` columns arrive as `TIMESTAMP` under Parquet autodetect. Use
  `--source_format=NEWLINE_DELIMITED_JSON` with `<table>.schema.json` if the
  distinction matters.
- `dim_stop_times` has four `INTERVAL` columns, which Parquet cannot carry.
  They are written as BigQuery interval literals (`0-0 0 6:2:0`) in `STRING`
  columns; cast with `CAST(col AS INTERVAL)` if you need the real type. No
  template reads one. Its `NUMERIC` columns are left null for the same reason.
