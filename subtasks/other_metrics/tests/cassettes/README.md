# Cassettes

Seven synthetic `mart_gtfs` datasets, one directory each. `conftest.py` stands a
DuckDB database up over one of them and runs the loaders' SQL against it
verbatim, so a cassette answers the query the loader actually issued.

A test's whole job is then: pick a cassette, and say what
`src/metric_calculation.py` should return for it. Nothing else is asserted — the
run goes from parquet straight to a reported table, and `src/data_loading.py` is
private as far as the tests are concerned, so it can change freely as long as the
metrics still come out the same.

One file per metric: `../test_travel_time_variability.py`,
`../test_expected_wait.py` and `../test_on_time_performance.py`.

| Directory | What it pins down |
| --- | --- |
| `constant_headway/` | service running exactly to schedule: excess wait is zero |
| `variable_headway/` | same mean headway, spread two different ways |
| `variable_travel_time/` | a p80/p20 travel time ratio of exactly 2.0 |
| `two_service_dates/` | headways either side of a service date boundary |
| `inconsistent_feed/` | a feed contradicting itself, and trip ids reused across dates |
| `mixed_punctuality/` | trips late all the way down the line, and trips early at a single stop |
| `punctuality_thresholds/` | trips landing exactly on the on time thresholds, and a second past them |

## What is in a cassette

Only the columns the templates in `../../templates/` actually read, at the
warehouse's types. Nothing does `select *`, so a column we never query is inert —
mirroring the warehouse's full column list would be a frozen copy rotting out of
date, with nothing to detect the drift and no test depending on it either way.

**A field added or dropped upstream that we do not query cannot reach these
tests.** What that leaves uncovered is a column we *do* query being renamed or
removed upstream: the cassette still has it, so the tests stay green while
production breaks. No local fixture can catch that — it needs a contract check
against the real warehouse.

`TABLES` at the top of `generate.py` is the list, and is the single place to
change when a template starts reading a new column.

## The shared pattern

Every cassette runs one feed (`feed1`) on one shape (`shape1`), over five stops
`stop1`–`stop5`. Stops 1, 2 and 5 are timepoints on every trip — the templates
only accept a stop as a timepoint where it is flagged on all of them — which
leaves two measured segments of different length and two stops that are not
timepoints at all, so the filter has something to drop.

Run time is 120 s stop to stop with a 30 s dwell, so a trip takes 600 s end to
end:

| quantity | value |
| --- | --- |
| travel time into stop 2 | 120 s |
| travel time into stop 5 | 420 s |
| headway on 15 minute service | 900 s |

A headway is the gap between vehicles — one departure to the next — so the dwell
drops out of it and 15 minute service publishes the full 900 s. The dwell is kept
non-zero anyway: it is the only thing separating an arrival from a departure, so
a metric reaching for the wrong one of the two shows up in the result.

There is **no jitter**: every vehicle arrives exactly when the scenario says it
does, so the expected metric values are known up front rather than measured off
the fixture.

## The scheduled baseline

`dim_stop_arrivals` publishes the same even 15 minute grid from 06:00 by default,
on that cassette's own `trip_id`s — sharing the ids is what lets a realtime stop
time find its scheduled counterpart. So the scheduled headway is 900 s, and a
cassette that runs something else against it has a measurable excess wait.

A scenario setting `scheduled_trips` opts out and publishes those instead. Three
do:

- `two_service_dates`, because where its seconds are counted from is the whole
  point of it
- `mixed_punctuality` and `punctuality_thresholds`, which publish the timetable
  their trips were built from and then move some of the trips off it. Against
  the shared 06:00 grid every trip would be hours out and no deviation would
  mean anything; against their own timetable a trip that was not moved deviates
  by exactly zero, and one that was deviates by exactly what it was moved.

It publishes both an `arrival_sec` and a `departure_sec`, separated by the dwell.
A headway runs departure to departure, so the wait metrics only ever read the
second; on time performance compares the published **arrival** against the one
the vehicle actually made, so a metric reaching for the wrong column of the two
comes out 30 s off.

Both schedule tables are grained by feed version rather than service date, so a
`trip_id` running on more than one date is published **once**. Publishing it
twice would fan the schedule join out and double every realtime stop time it
matched. Only `inconsistent_feed` reuses an id, but the rule is in
`build_tables` for all of them.

## constant_headway

8 trips, 06:00 to 07:45, exactly to schedule.

| | |
| --- | --- |
| gaps per stop | 7, all 900 s |
| day parts | 4 in Early AM, 3 in AM Peak, at every stop |
| travel time variability | 1.0 on both segments |
| expected wait | 450 s, scheduled and realtime alike |
| excess wait | 0 |

## variable_headway

24 trips. 07:00–10:00 alternates 10 and 20 minute gaps; 10:00–13:00 runs an even
15. Alternating 10 and 20 averages exactly 15, so the two blocks report the same
mean headway, the same median and the same schedule, and differ only in how
evenly service was spread. A rider does not experience them as the same — they
are twice as likely to arrive during a 20 minute gap as a 10 minute one — and
that difference, with nothing else moving alongside it, is what expected wait
exists to measure.

At every stop:

| day part | gaps | mean | variance | E[wait] | excess | % time over-long |
| --- | --- | --- | --- | --- | --- | --- |
| AM Peak | 6 × 600 s, 6 × 1200 s | 900 s | 98181.82 | 504.55 s | +54.55 s | 0.166667 |
| Midday | 11 × 900 s | 900 s | 0 | 450.00 s | 0 | 0 |

The last column counts a rider's own wait, not the gap's length. In a 1200 s gap
against a scheduled 900, only riders arriving in the first 300 seconds wait
longer than the schedule promised. So each long gap contributes `1200 - 900`,
giving `6 × 300 / 10800 = 1/6`.

A trip takes 10 minutes to run the line, so both blocks sit clear of their day
part's boundary at every stop: the table above holds identically at stop 1 and at
stop 5. The gap handing over between blocks opens at 09:40, inside AM Peak, and
is 20 minutes — the alternating cycle's own long step, so it belongs to that
block's pattern rather than distorting it.

## variable_travel_time

9 trips at 15 minutes, every third one running the whole line in half the time.
Travel time takes exactly two values, so p80/p20 is exactly **2.0**:

| segment | p80 | p20 | ratio |
| --- | --- | --- | --- |
| into stop 2 | 120 s | 60 s | 2.0 |
| into stop 5 | 420 s | 210 s | 2.0 |

### Why one third

For `p80/p20` to equal `slow/fast`, p20 has to land inside the fast group and p80
inside the slow one. Under pandas' linear interpolation over `n` trips that needs

```
0.2 + 0.8/n  <=  f  <=  0.8 - 0.8/n
```

`f = 0.2` fails the lower bound at every `n` — a 20/80 split gives 1.11, not 2.0.
Worse, no choice of fast travel time rescues it: with only 20% fast, p20 stays
inside the slow group, capping the ratio at `1/0.8 = 1.25`. One third clears the
bound for any `n` of 6 or more; 9 trips gives 3 fast and 6 slow, with a whole trip
of room either side.

Fast trips run down the line quicker than slow ones, so past stop 1 the headway
here is no longer even. **Use this cassette for travel time variability, not for
headway.**

## two_service_dates

8 trips over a date boundary:

- **`2026-06-17`, 23:30–24:15 departures** — running on past midnight
- **`2026-06-18`, 04:00–04:45 departures** — listed under the next date

Each date runs its own trips, as a real feed would.

**GTFS hours run past 24.** 6/17's last trips depart at 24:00 and 24:15. They are
still 6/17's service, and the schedule publishes them as hour 24, while the
realtime side records the same trips as ordinary datetimes on 6/18:

| trip departs | `departure_sec` | `service_date` | `actual_departure_pacific` |
| --- | --- | --- | --- |
| `23:45` | 85530 | 2026-06-17 | 2026-06-17 23:45:30 |
| `24:00` | 86430 | 2026-06-17 | 2026-06-18 00:00:30 |
| `04:00` | 14430 | 2026-06-18 | 2026-06-18 04:00:30 |

**A trip's service date is what it is listed under, not what the clock says.**
6/18's service does not start until 04:00, and its seconds are counted from
6/18's midnight — so the trip that runs *later* is published as the *smaller*
number of seconds, and the count restarts mid-window. Anything subtracting one
from the other has to say which midnight it means.

`schedule_rt_headways.sql` answers that by pairing within a service date, so:

| | |
| --- | --- |
| gaps per stop | 6 — 3 on each date, all 900 s |
| the overnight break | not reported as a headway at all |
| the 23:45 → 24:00 pair | still 900 s: crossing midnight is not crossing a service date |
| day parts | 2 Evening, 1 Owl, 3 Early AM |
| `n_service_dates` | 2 — the segment grain pools both dates |

Owl holds a single gap, which has no sample variance, so its realtime expected
wait comes back `NaN` rather than half the one gap. That is deliberate behaviour
in `calculate_expected_wait`, and this is the cassette that exercises it.

## inconsistent_feed

Every other cassette is a well formed feed. This one is not, in two ways that
real feeds are.

**Trip ids are reused across dates.** Four ids — `daily-00` to `daily-03` — run
at 06:00 to 06:45 on both `2026-06-17` and `2026-06-18`. That is 8 runs under 4
ids, which is what a real feed looks like and what a grain missing
`service_date` would silently collapse to 4.

**The feed contradicts itself about the order of two stops.** `daily-01`, on
each date, reports reaching stop 2 ten seconds after it reached stop 1 — 20 s
*before* it departed stop 1, and 140 s earlier than the 150 s the run takes.
Not a fast trip, an impossible one. Its departure from stop 2 is left where it
was, so only the one segment is affected.

| | |
| --- | --- |
| gaps per stop | 6 — 3 on each date, none pairing across the overnight break |
| trips per segment | 8, on 2 service dates |
| travel time into stop 2 | 2 of 8 negative and dropped; the other 6 are 120 s |
| travel time into stop 5 | 420 s on all 8 — the disorder does not carry forward |

Dropping the negatives is what keeps p20 above zero: 2 negatives out of 8 puts
the 20th percentile inside them, and the segment would report a ratio well
under 1 for a run every real trip made in the same time. The trips themselves
are still counted — `n_rt_trips` is 8 over 6 usable travel times.

The wait metric never sees the contradiction. A headway runs departure to
departure, and only the arrival was wrong, so every stop here reports the even
service it ran — 900 s gaps, 450 s expected wait, no excess.

That is the definition earning its keep. Measuring to the next arrival instead
would have taken the bad stop time at face value and reported a 730 s gap where
the two vehicles were a clean 900 s apart, giving stop 2 a *negative* excess
wait: a feed error surfacing as service that beat its timetable.

## mixed_punctuality

10 trips every 15 minutes from 04:15, measured against the timetable they were
built from. Four run exactly to time. Three run 360 s late at every stop. Three
run to time all the way down the line and then reach the **last** timepoint
120 s early, their departures left where they were.

| stop | deviations | p50 \|dev\| | p80 \|dev\| | early | late |
| --- | --- | --- | --- | --- | --- |
| 1 and 2 | 7 × 0 s, 3 × +360 s | 0 s | 360 s | 0 | 0.3 |
| 5 | 4 × 0 s, 3 × −120 s, 3 × +360 s | 120 s | 360 s | 0.3 | 0.3 |
| pooled | 18 × 0 s, 3 × 120 s, 9 × 360 s | 0 s | 360 s | 3/30 | 9/30 |

The asymmetry is the point. A trip late everywhere and a trip early in one place
cannot both be reported by a per-trip number, so the last timepoint has to differ
from the two before it — and the pooled row cannot be the average of the three.
It is also the only cassette where per-arrival and per-trip counting disagree:
counting trips would give 3/10 and 3/10 pooled rather than 3/30 and 9/30.

Three of each rather than one of each so the percentiles separate the stops as
well as the proportions do — with a single early trip the last timepoint's median
stayed on zero with everything else.

Everything falls inside Early AM, so the day part never enters into it. The early
arrivals do shorten one segment, so the travel time into stop 5 takes two values
here: **use this cassette for punctuality, not travel time.**

## punctuality_thresholds

8 trips every 15 minutes from 04:30. A whole trip moves together, so each carries
one deviation the whole way down the line and every timepoint reports the same
split:

| trip | deviation | early? | late? |
| --- | --- | --- | --- |
| 1 | −60 s | **yes** — exactly on the threshold | |
| 2 | −61 s | **yes** | |
| 3 | +300 s | | **yes** — exactly on the threshold |
| 4 | +301 s | | **yes** |
| the other four | 0 s | | |

Both comparisons are inclusive, so half the service misses its published time
and a quarter of it is reported as early, a quarter as late. **A `<` in place of
a `<=` halves both proportions and nothing else in the suite notices.**

It is also the only cassette whose percentiles land inside an interpolation
rather than on a repeated value, which is what makes the two tables visibly
answer different questions:

| | sorted \|deviation\| | p50 | p80 |
| --- | --- | --- | --- |
| per stop | 0, 0, 0, 0, 60, 61, 300, 301 | 30.0 | 61 + 0.6 × 239 = **204.4** |
| pooled | the same eight, three times over | 30.0 | **300.0** |

All three stops report 204.4, so any averaging of them gives 204.4 back. Pooling
the 24 arrivals instead puts the 80th percentile at the 18.4th sorted value,
inside the three 300s.

Every stop time of a trip moves by the same amount, so travel times are
untouched and this cassette stays in the constant travel time set.

## Regenerating

Parquet is gitignored, so generate before running the tests:

```bash
uv run python tests/cassettes/generate.py
```

Constants at the top of `generate.py` control the stop pattern, timepoints, run
time and dwell; each scenario function controls its own trips. Everything is
fully deterministic — there is no RNG.
