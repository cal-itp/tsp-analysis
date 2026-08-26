# Cassettes

Four synthetic `mart_gtfs` datasets, one directory each. `conftest.py` stands a
DuckDB database up over one of them and runs the loaders' SQL against it
verbatim, so a cassette answers the query the loader actually issued.

A test's whole job is then: pick a cassette, and say what
`src/metric_calculation.py` should return for it. Nothing else is asserted — the
run goes from parquet straight to a reported table, and `src/data_loading.py` is
private as far as the tests are concerned, so it can change freely as long as the
metrics still come out the same.

One file per metric: `../test_travel_time_variability.py` and
`../test_expected_wait.py`.

| Directory | What it pins down |
| --- | --- |
| `constant_headway/` | service running exactly to schedule: excess wait is zero |
| `variable_headway/` | same mean headway, spread two different ways |
| `variable_travel_time/` | a p80/p20 travel time ratio of exactly 2.0 |
| `two_service_dates/` | headways either side of a service date boundary |
| `inconsistent_feed/` | a feed contradicting itself, and trip ids reused across dates |

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
| headway on 15 minute service | 870 s |

The headway is 870 rather than 900 because a rider's gap runs from one vehicle's
departure to the next one's arrival, so it is one dwell short of the spacing.
Keeping the dwell non-zero is what makes a departure/arrival mix-up visible.

There is **no jitter**: every vehicle arrives exactly when the scenario says it
does, so the expected metric values are known up front rather than measured off
the fixture.

## The scheduled baseline

`dim_stop_arrivals` publishes the same even 15 minute grid from 06:00 in every
cassette but one, on that cassette's own `trip_id`s — sharing the ids is what
lets a realtime stop time find its scheduled counterpart. So the scheduled
headway is 870 s everywhere, and a cassette that runs something else against it
has a measurable excess wait.

`two_service_dates` opts out and publishes each trip's own times instead: where
those seconds are counted from is the whole point of it.

Both schedule tables are grained by feed version rather than service date, so a
`trip_id` running on more than one date is published **once**. Publishing it
twice would fan the schedule join out and double every realtime stop time it
matched. Only `inconsistent_feed` reuses an id, but the rule is in
`build_tables` for all of them.

## constant_headway

8 trips, 06:00 to 07:45, exactly to schedule.

| | |
| --- | --- |
| gaps per stop | 7, all 870 s |
| day parts | 4 in Early AM, 3 in AM Peak, at every stop |
| travel time variability | 1.0 on both segments |
| expected wait | 435 s, scheduled and realtime alike |
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
| AM Peak | 6 × 570 s, 6 × 1170 s | 870 s | 98181.82 | 491.43 s | +56.43 s | 0.172414 |
| Midday | 11 × 870 s | 870 s | 0 | 435.00 s | 0 | 0 |

The last column counts a rider's own wait, not the gap's length. In a 1170 s gap
against a scheduled 870, only riders arriving in the first 300 seconds wait
longer than the schedule promised. So each long gap contributes `1170 - 870`,
giving `6 × 300 / 10440 = 0.172414`.

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

| trip departs | `arrival_sec` | `service_date` | `actual_arrival_pacific` |
| --- | --- | --- | --- |
| `23:45` | 85500 | 2026-06-17 | 2026-06-17 23:45:00 |
| `24:00` | 86400 | 2026-06-17 | 2026-06-18 00:00:00 |
| `04:00` | 14400 | 2026-06-18 | 2026-06-18 04:00:00 |

**A trip's service date is what it is listed under, not what the clock says.**
6/18's service does not start until 04:00, and its seconds are counted from
6/18's midnight — so the trip that runs *later* is published as the *smaller*
number of seconds, and the count restarts mid-window. Anything subtracting one
from the other has to say which midnight it means.

`schedule_rt_headways.sql` answers that by pairing within a service date, so:

| | |
| --- | --- |
| gaps per stop | 6 — 3 on each date, all 870 s |
| the overnight break | not reported as a headway at all |
| the 23:45 → 24:00 pair | still 870 s: crossing midnight is not crossing a service date |
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

The wait metric has no such filter and cannot have one, because a short gap is
not on its face a wrong one. So the bad arrival lands in the headways as a
genuine 730 s gap (870 − 140), and stop 2 reports a wait *better* than the
schedule promised:

| | mean | variance | E[wait] | excess |
| --- | --- | --- | --- | --- |
| stop 2 | 823.33 s | 5226.67 | 414.84 s | **−20.16 s** |
| every other stop | 870 s | 0 | 435 s | 0 |

Excess wait goes negative and nothing clamps it, which is the honest answer: a
metric that floored it at zero would report the same figure here as for service
that ran exactly to time. `percent_time_excess_wait` stays 0 — it counts long
gaps, and there are none.

## Regenerating

Parquet is gitignored, so generate before running the tests:

```bash
uv run python tests/cassettes/generate.py
```

Constants at the top of `generate.py` control the stop pattern, timepoints, run
time and dwell; each scenario function controls its own trips. Everything is
fully deterministic — there is no RNG.
