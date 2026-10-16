"""Generate the synthetic mart_gtfs cassettes the metric tests play against.

One directory per scenario, one parquet file per warehouse table the templates
in ../../templates/ name. Each table carries **only the columns those templates
read**, at the warehouse's types. Nothing does ``select *``, so a column we never
query is inert -- mirroring the warehouse's full column list would only be a
frozen copy rotting out of date, with nothing to detect the drift and no test
depending on it either way. A field added or dropped upstream that we do not
query therefore cannot reach these tests at all.

What that leaves uncovered is a column we *do* query being renamed or removed
upstream. No local fixture can catch that; it needs a contract check against the
real warehouse.

Every scenario shares one feed, one shape and one 5 stop pattern, and runs
without jitter: each vehicle arrives exactly when the scenario says it does. So
the value each metric should report is known up front rather than measured off
the fixture.

    constant_headway/       a trip every 15 minutes, running exactly to schedule
    variable_headway/       alternating 10/20 through AM Peak, an even 15 through
                            Midday -- same mean, same schedule, different spread
    variable_travel_time/   every third trip runs the line in half the time, so
                            the p80/p20 ratio is exactly 2.0
    two_service_dates/      service either side of a date boundary, published in
                            GTFS hours past 24
    inconsistent_feed/      a feed contradicting itself about the order of two
                            stops, and one trip id running on two dates
    mixed_punctuality/      ten trips against their own timetable: three late all
                            the way down the line, three early at one stop only
    punctuality_thresholds/ trips landing exactly on the on time thresholds, and
                            one second past them

Run with the project venv:

    uv run python tests/cassettes/generate.py
"""

from __future__ import annotations

import datetime
import itertools
import pathlib
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# The tables, as the templates read them
# ---------------------------------------------------------------------------

# BigQuery has no parquet equivalent for GEOGRAPHY, so it travels as WKT the way
# the warehouse exports it. DATETIME is naive, TIMESTAMP is UTC, as upstream
WKT = pa.string()

TABLES = {
    # shapes_routes.sql: which shapes ran the target route on each date
    "fct_scheduled_trips": pa.schema(
        [
            ("service_date", pa.date32()),
            ("feed_key", pa.string()),
            ("trip_id", pa.string()),
            ("shape_id", pa.string()),
            ("direction_id", pa.int64()),
            ("route_id", pa.string()),
            ("route_short_name", pa.string()),
        ]
    ),
    # the realtime side of both metrics: when each vehicle actually reached each stop
    "fct_stop_time_metrics": pa.schema(
        [
            ("service_date", pa.date32()),
            ("schedule_base64_url", pa.string()),
            ("trip_id", pa.string()),
            ("trip_key", pa.string()),
            ("stop_id", pa.string()),
            ("stop_sequence", pa.int64()),
            ("actual_arrival_pacific", pa.timestamp("us")),
            ("actual_departure_pacific", pa.timestamp("us")),
            ("n_predictions", pa.int64()),
        ]
    ),
    # the only table exposing the timepoint flag
    "dim_stop_times": pa.schema(
        [
            ("feed_key", pa.string()),
            ("trip_id", pa.string()),
            ("stop_id", pa.string()),
            ("stop_sequence", pa.int64()),
            ("timepoint", pa.int64()),
            ("_feed_valid_from", pa.timestamp("us", tz="UTC")),
        ]
    ),
    # the published schedule, in seconds since midnight on the trip's service
    # date. The headway metric reads only the departure -- a headway is measured
    # departure to departure -- while on time performance compares the published
    # arrival against the one the vehicle actually made
    "dim_stop_arrivals": pa.schema(
        [
            ("feed_key", pa.string()),
            ("trip_id", pa.string()),
            ("stop_id", pa.string()),
            ("stop_sequence", pa.int64()),
            ("arrival_sec", pa.int64()),
            ("departure_sec", pa.int64()),
            ("_feed_valid_from", pa.timestamp("us", tz="UTC")),
        ]
    ),
    # stop geometry, so a metric can be mapped
    "fct_daily_scheduled_stops": pa.schema(
        [
            ("service_date", pa.date32()),
            ("feed_key", pa.string()),
            ("stop_id", pa.string()),
            ("pt_geom", WKT),
        ]
    ),
    # feed_key -> the base64_url and _valid_from that partition the stop times tables
    "dim_schedule_feeds": pa.schema(
        [
            ("key", pa.string()),
            ("base64_url", pa.string()),
            ("_valid_from", pa.timestamp("us", tz="UTC")),
        ]
    ),
    # which feed was in effect on which date
    "fct_daily_schedule_feeds": pa.schema(
        [
            ("date", pa.timestamp("us")),
            ("feed_key", pa.string()),
            ("gtfs_dataset_name", pa.string()),
        ]
    ),
    # shape geometry
    "fct_daily_scheduled_shapes": pa.schema(
        [
            ("service_date", pa.date32()),
            ("feed_key", pa.string()),
            ("shape_id", pa.string()),
            ("pt_array", pa.list_(WKT)),
        ]
    ),
}


# ---------------------------------------------------------------------------
# The one feed, shape and stop pattern every scenario runs on
# ---------------------------------------------------------------------------

OUT_ROOT = pathlib.Path(__file__).parent

FEED_KEY = "feed1"
SHAPE_ID = "shape1"
ROUTE_ID = "route1"
ROUTE_SHORT_NAME = "R1"
GTFS_DATASET_NAME = "Test Schedule"

# base64_url is only ever a join key between the schedule and realtime sides --
# nothing decodes it, so it is spelled out rather than encoded
SCHEDULE_BASE64_URL = "schedule-url"

# partitions dim_stop_times and dim_stop_arrivals, and is what the loaders carry
# from dim_schedule_feeds through to the stop times queries
FEED_VALID_FROM = datetime.datetime(2026, 6, 16, tzinfo=datetime.UTC)

STOP_IDS = ["stop1", "stop2", "stop3", "stop4", "stop5"]

# stops 1, 2 and 5 are timepoints, on every trip -- the templates only accept a
# stop as a timepoint where it is flagged on all of them. That leaves two
# measured segments of deliberately different length, 1 -> 2 and 2 -> 5, and two
# stops that are not timepoints at all, so the filter has something to drop
TIMEPOINT_SEQUENCES = (1, 2, 5)

# stop to stop run time and dwell, both in seconds. A trip therefore takes
# 4 * (120 + 30) = 600 s end to end, the travel time into stop 2 is 120 s, and
# the travel time into stop 5 is 420 s
SEGMENT_SECONDS = 120
DWELL_SECONDS = 30

# a headway is the gap between vehicles, one departure to the next, so 15 minute
# service publishes the full 900 s and the dwell drops out of it. The dwell is
# kept non-zero anyway: it is what separates an arrival from a departure, so a
# metric reaching for the wrong one of the two shows up in the result
HEADWAY_SECONDS = 15 * 60


def midnight(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.time())


def seconds_since_midnight(when: datetime.datetime, service_date: datetime.date) -> int:
    """A stop time as GTFS counts it: from midnight on its own service date.

    The count restarts on each service date, so a trip running past midnight
    keeps counting up into hours 24 and beyond rather than wrapping.
    """
    return int((when - midnight(service_date)).total_seconds())


def stop_point(sequence: int) -> str:
    """WKT for a stop, spaced along a corridor so the geometry is a real line."""
    return f"POINT(-122.{sequence:02d}0000 37.500000)"


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------


@dataclass
class Trip:
    """One vehicle trip: when it reached each stop, and the date it is listed under."""

    trip_id: str
    service_date: datetime.date
    # (stop_sequence, stop_id, arrival, departure), pacific and naive, as the
    # warehouse's *_pacific columns are
    stop_times: list[tuple[int, str, datetime.datetime, datetime.datetime]]

    @property
    def trip_key(self) -> str:
        # fct_stop_time_metrics builds this from the realtime feed, so it is not
        # the schedule side's key. It breaks ties when two vehicles reach a stop
        # at the same moment, so it has to sort in the order the trips ran -- and
        # it is unique per run, so a trip_id running on two dates gets two keys
        return f"{self.service_date.isoformat()}-{self.trip_id}-rt"


@dataclass
class Scenario:
    name: str
    trips: list[Trip]
    # dim_stop_arrivals normally publishes the even 15 minute grid in
    # build_schedule, which is what makes an excess wait measurable where the
    # service departs from it. A scenario that needs its schedule to be something
    # else -- its own trips, or the timetable those trips then ran late against --
    # hands the published trips over here instead
    scheduled_trips: list[Trip] | None = None
    notes: list[str] = field(default_factory=list)


def build_trip(
    trip_id: str,
    service_date: datetime.date,
    start: datetime.datetime,
    factor: float = 1,
) -> Trip:
    """A trip reaching stop 1 at ``start``, running the pattern at ``factor`` speed.

    ``factor`` scales the dwell as well as the run time -- a fast trip that kept
    the full dwell would not be a clean multiple of a slow one, and the unscaled
    dwell would survive into every timepoint-to-timepoint difference.
    """
    stop_times = []
    cursor = start
    for sequence, stop_id in enumerate(STOP_IDS, start=1):
        arrival = cursor
        departure = arrival + datetime.timedelta(seconds=DWELL_SECONDS * factor)
        stop_times.append((sequence, stop_id, arrival, departure))
        cursor = departure + datetime.timedelta(seconds=SEGMENT_SECONDS * factor)
    return Trip(trip_id=trip_id, service_date=service_date, stop_times=stop_times)


def disorder_arrival(trip: Trip, sequence: int, seconds_early: int) -> Trip:
    """Pull one arrival back before the departure from the stop before it.

    A feed contradicting itself about the order of two stops, which is what
    produces a negative travel time. The departure from the disordered stop is
    left where it was, so only the one segment is affected and everything after
    it still runs to time -- the trip is not late, the feed is wrong.
    """
    stop_times = list(trip.stop_times)
    index = next(
        i for i, stop_time in enumerate(stop_times) if stop_time[0] == sequence
    )
    prior_departure = stop_times[index - 1][3]
    _, stop_id, _, departure = stop_times[index]
    stop_times[index] = (
        sequence,
        stop_id,
        prior_departure - datetime.timedelta(seconds=seconds_early),
        departure,
    )
    return Trip(
        trip_id=trip.trip_id, service_date=trip.service_date, stop_times=stop_times
    )


def shift_trip(trip: Trip, seconds: int) -> Trip:
    """Run a whole trip late, stop for stop. Negative seconds run it early.

    Every stop time moves by the same amount, so the trip is late everywhere and
    its travel time between any two stops is untouched -- a late bus is late, not
    slow. That is what separates this from ``build_trip(factor=...)``, which
    changes how long the run takes and leaves the start where it was.
    """
    offset = datetime.timedelta(seconds=seconds)
    return Trip(
        trip_id=trip.trip_id,
        service_date=trip.service_date,
        stop_times=[
            (sequence, stop_id, arrival + offset, departure + offset)
            for sequence, stop_id, arrival, departure in trip.stop_times
        ],
    )


def shift_arrival(trip: Trip, sequence: int, seconds: int) -> Trip:
    """Move one arrival, leaving its departure and every other stop alone.

    A bus reaching one stop early and leaving it on time, rather than running the
    whole line early. Only the one stop's punctuality moves.
    """
    offset = datetime.timedelta(seconds=seconds)
    return Trip(
        trip_id=trip.trip_id,
        service_date=trip.service_date,
        stop_times=[
            (
                stop_sequence,
                stop_id,
                arrival + offset if stop_sequence == sequence else arrival,
                departure,
            )
            for stop_sequence, stop_id, arrival, departure in trip.stop_times
        ],
    )


def distinct_trips(trips: list[Trip]) -> list[Trip]:
    """One trip per trip_id, in first-appearance order.

    dim_stop_times and dim_stop_arrivals are grained by feed version rather than
    service date, so a trip_id running on more than one date is published exactly
    once -- publishing it twice would fan the schedule join out and double every
    realtime stop time it matched.
    """
    by_id: dict[str, Trip] = {}
    for trip in trips:
        by_id.setdefault(trip.trip_id, trip)
    return list(by_id.values())


def even_starts(
    first: datetime.datetime, headway_minutes: int, count: int
) -> list[datetime.datetime]:
    return [
        first + datetime.timedelta(minutes=headway_minutes * t) for t in range(count)
    ]


def cycled_starts(
    first: datetime.datetime, until: datetime.datetime, cycle: tuple[int, ...]
) -> list[datetime.datetime]:
    """Trips from ``first``, walking ``cycle`` round in minutes until ``until``.

    The gap carrying the end of a block belongs to the outgoing block: a block
    ending on a 20 minute step hands over 20 minutes after its last trip.
    """
    starts = []
    cursor = first
    for step in itertools.count():
        if cursor >= until:
            break
        starts.append(cursor)
        cursor += datetime.timedelta(minutes=cycle[step % len(cycle)])
    return starts


DATE = datetime.date(2026, 6, 17)
NEXT_DATE = datetime.date(2026, 6, 18)

# inconsistent_feed's contradiction: the second trip of each date, and the stop
# it reports reaching before it left the one before. Stop 2 is a timepoint, so
# the bad travel time reaches the metric rather than being filtered out with the
# non-timepoints
DISORDERED_TRIP = 1
DISORDERED_SEQUENCE = 2
SECONDS_EARLY = 20


def constant_headway() -> Scenario:
    """A trip every 15 minutes, running exactly to the published schedule.

    06:00 to 07:45 departures, so the gaps split across Early AM and AM Peak the
    same way at every stop. Every gap is 900 s and so is every scheduled gap, so
    the excess wait is zero and the expected wait is half a headway exactly.
    """
    starts = even_starts(midnight(DATE) + datetime.timedelta(hours=6), 15, 8)
    return Scenario(
        name="constant_headway",
        trips=[
            build_trip(f"constant-{t:02d}", DATE, start)
            for t, start in enumerate(starts)
        ],
        notes=[
            "8 trips, 7 gaps per stop, all 900 s",
            "4 gaps in Early AM and 3 in AM Peak, at every stop",
            "travel time 120 s into stop 2 and 420 s into stop 5, on every trip",
        ],
    )


def variable_headway() -> Scenario:
    """The same mean headway run two different ways, in two different day parts.

    07:00 to 10:00 alternates 10 and 20 minute gaps; 10:00 to 13:00 runs an even
    15. Alternating 10 and 20 averages exactly 15, so the two blocks report the
    same mean headway, the same median and the same schedule, and differ only in
    how evenly the service was spread -- which is the whole of what expected wait
    measures.

    A trip takes 10 minutes to run the line, so both blocks sit clear of their
    day part's boundary at every stop: AM Peak sees only alternating gaps and
    Midday only even ones, from stop 1 through stop 5.

    The gap handing over from one block to the next opens at 09:40, inside AM
    Peak, and is 20 minutes -- which is the alternating cycle's own long step, so
    it belongs to that block's pattern rather than distorting it.
    """
    starts = cycled_starts(
        midnight(DATE) + datetime.timedelta(hours=7),
        midnight(DATE) + datetime.timedelta(hours=10),
        (10, 20),
    ) + even_starts(midnight(DATE) + datetime.timedelta(hours=10), 15, 12)
    return Scenario(
        name="variable_headway",
        trips=[
            build_trip(f"variable-{t:02d}", DATE, start)
            for t, start in enumerate(starts)
        ],
        notes=[
            "24 trips, 23 gaps per stop",
            "AM Peak: 6 gaps of 600 s and 6 of 1200 s, at every stop",
            "Midday: 11 gaps of 900 s, at every stop",
            "both day parts: mean 900 s, median 900 s, scheduled 900 s",
        ],
    )


def variable_travel_time() -> Scenario:
    """Every third trip runs the whole line in half the time, so p80/p20 is 2.0.

    The fraction is not free. For p80/p20 to equal slow/fast, p20 has to land
    inside the fast group and p80 inside the slow one, which under pandas' linear
    interpolation over n trips needs ``0.2 + 0.8/n <= f <= 0.8 - 0.8/n``. A 20/80
    split fails the lower bound at every n -- and no choice of fast travel time
    rescues it, because p20 stays inside the slow group and caps the ratio at
    1.25. One third clears the bound for any n of 6 or more; 9 trips gives 3 fast
    and 6 slow, with a whole trip of room either side.

    Fast trips run down the line quicker than slow ones, so past stop 1 the
    headway here is no longer even. Use this scenario for travel time
    variability, not for headway.
    """
    starts = even_starts(midnight(DATE) + datetime.timedelta(hours=6), 15, 9)
    return Scenario(
        name="variable_travel_time",
        trips=[
            build_trip(f"travel-{t:02d}", DATE, start, factor=0.5 if t % 3 == 0 else 1)
            for t, start in enumerate(starts)
        ],
        notes=[
            "9 trips, 3 of them running every segment in half the time",
            "travel time into stop 2: p80 120 s, p20 60 s, ratio 2.0",
            "travel time into stop 5: p80 420 s, p20 210 s, ratio 2.0",
        ],
    )


def two_service_dates() -> Scenario:
    """Service either side of a date boundary, published in GTFS hours past 24.

    Every other scenario covers a single date and stops well short of midnight,
    which hides two things.

    GTFS hours run past 24. 6/17's last trips depart at 24:00 and 24:15 -- still
    6/17's service, published as hour 24, while the realtime side records them as
    ordinary datetimes on 6/18. A stop time's GTFS hour and its clock hour are
    not the same number, and only the schedule side counts in GTFS hours.

    And a trip's service date is what it is listed under, not what the clock
    says. 6/18's own service starts at 04:00, hours after 6/17's finished, and
    its seconds are counted from 6/18's midnight -- so the trip that runs later is
    published as the smaller number of seconds, and the count restarts mid-window.

    Pairing within a service date is what keeps the two sides of a headway
    counted from the same midnight: the overnight break is not reported at all,
    while 6/17's 23:45 and 24:00 trips still pair at the usual 900 s, because
    crossing midnight is not the same thing as crossing a service date.
    """
    trips: list[Trip] = []
    for date, first_hour in [(DATE, 23.5), (NEXT_DATE, 4)]:
        starts = even_starts(
            midnight(date) + datetime.timedelta(hours=first_hour), 15, 4
        )
        trips += [
            build_trip(f"overnight-{len(trips) + t:02d}", date, start)
            for t, start in enumerate(starts)
        ]
    return Scenario(
        name="two_service_dates",
        trips=trips,
        # the point of this scenario is where the published seconds are counted
        # from, so it publishes its own times rather than the shared grid
        scheduled_trips=trips,
        notes=[
            "8 trips, 4 on each date, 3 gaps per stop per date",
            "the overnight break between the dates is not reported as a headway",
            "6/17's gaps open 2 in Evening and 1 in Owl, 6/18's 3 in Early AM",
            "Owl holds a single gap, so its realtime expected wait is NaN",
            "every gap 900 s, scheduled and realtime alike",
        ],
    )


def inconsistent_feed() -> Scenario:
    """A feed contradicting itself, and one set of trip ids running on two dates.

    Every other scenario is a well formed feed, which leaves two things untested.

    **A trip_id is only unique within a service date.** Real feeds reuse them
    daily, and every other scenario gives each run its own id, so nothing catches
    a grain that has dropped ``service_date``. Here the same four ids run on both
    dates: 8 runs under 4 ids, which must be counted as 8 trips and paired into
    3 gaps on each date rather than 7 across the pair. Note what the schedule
    tables do with that -- ``dim_stop_arrivals`` is grained by feed version, not
    by date, so it publishes each id exactly once and both days' runs measure
    themselves against that one published time.

    **A feed can disagree with itself about the order of two stops.** The second
    trip of each date reports reaching stop 2 twenty seconds *before* it left
    stop 1, which is not a fast trip but an impossible one. Its travel time is
    negative, and dropping it is the only thing keeping the p20 above zero: with
    2 of the 8 observations negative, a p20 that kept them would interpolate
    somewhere below the fastest real trip and report a ratio well under 1. The
    trip itself is still real and still counted -- only the one travel time is
    discarded -- so the segment reports 8 trips over 6 usable travel times.

    The wait metric never sees the contradiction at all. A headway runs departure
    to departure, and only the arrival was wrong, so every stop here reports the
    even service it ran. Measuring to the next arrival instead would have taken
    the bad stop time at face value and reported a 730 s gap where the two
    vehicles were a clean 900 s apart.

    Everything else runs to the even 15 minute schedule, so the disorder is the
    only thing moving.
    """
    trips: list[Trip] = []
    for date in [DATE, NEXT_DATE]:
        starts = even_starts(midnight(date) + datetime.timedelta(hours=6), 15, 4)
        for t, start in enumerate(starts):
            trip = build_trip(f"daily-{t:02d}", date, start)
            if t == DISORDERED_TRIP:
                trip = disorder_arrival(
                    trip, sequence=DISORDERED_SEQUENCE, seconds_early=SECONDS_EARLY
                )
            trips.append(trip)
    return Scenario(
        name="inconsistent_feed",
        trips=trips,
        notes=[
            "8 runs under 4 trip ids, on 2 service dates",
            "3 gaps per stop per date, none pairing across the overnight break",
            "one trip per date reaches stop 2 before it left stop 1",
            "stop 2: 8 trips, 6 usable travel times of 120 s, p80/p20 still 1.0",
            "stop 5 is untouched, so its 420 s travel time holds on all 8",
            "the wait is unmoved: a headway runs departure to departure",
        ],
    )


# on time performance's two thresholds: a vehicle is on time unless it beats its
# published arrival by a minute or more, or misses it by five minutes or more.
# Stated here because the two punctuality scenarios are built to land either side
# of them, not because the generator applies them
EARLY_THRESHOLD_SECONDS = 60
LATE_THRESHOLD_SECONDS = 5 * 60

# mixed_punctuality: which of the ten trips misbehave, and how. Three of each,
# rather than one, so that a percentile has somewhere to land other than a
# repeated zero: with 3 of 10 arrivals off the mark, both the median and the 80th
# percentile of the last timepoint's absolute deviations move
LATE_TRIPS = (1, 4, 7)
LATE_SECONDS = 6 * 60
EARLY_ARRIVAL_TRIPS = (3, 6, 9)
EARLY_ARRIVAL_SECONDS = 2 * 60
LAST_TIMEPOINT = TIMEPOINT_SEQUENCES[-1]

# punctuality_thresholds: trip index -> how late it runs, in seconds. Two land
# exactly on a threshold and two land one second past it
BOUNDARY_DEVIATIONS = {
    1: -EARLY_THRESHOLD_SECONDS,
    2: -EARLY_THRESHOLD_SECONDS - 1,
    3: LATE_THRESHOLD_SECONDS,
    4: LATE_THRESHOLD_SECONDS + 1,
}


def mixed_punctuality() -> Scenario:
    """Ten trips against their own timetable: three late throughout, three early once.

    The schedule here is the timetable the trips were built from rather than the
    shared 06:00 grid, so a trip that ran to it deviates by exactly zero and the
    ones that did not deviate by exactly what they were moved.

    Four trips run to time. Three run six minutes late at every stop -- late
    enough to miss the five minute threshold, and late everywhere, so every
    timepoint reports them. Three run to time all the way down the line and then
    reach the *last* timepoint two minutes early, their departures left where
    they were: early enough to beat the one minute threshold, at one stop only.

    That asymmetry is the point. A metric reported per stop must show the early
    trips at the last timepoint and nowhere else, so the first two timepoints and
    the last one cannot report the same numbers -- and the pooled table cannot be
    the average of the per-stop ones.

    Three of each rather than one of each, so the percentiles separate the stops
    as well as the proportions do. Sorted absolute deviations are

        stops 1 and 2   0 x 7, 360 x 3          p50 =   0.0   p80 = 360.0
        stop 5          0 x 4, 120 x 3, 360 x 3 p50 = 120.0   p80 = 360.0

    With a single early trip the last timepoint's median stayed on zero with
    everything else, and only the proportions told the stops apart.

    Trips run every 15 minutes from 04:15, so all thirty timepoint arrivals fall
    inside Early AM and the day part is never what splits them.
    """
    starts = even_starts(
        midnight(DATE) + datetime.timedelta(hours=4, minutes=15), 15, 10
    )
    scheduled = [
        build_trip(f"mixed-{t:02d}", DATE, start) for t, start in enumerate(starts)
    ]

    ran = list(scheduled)
    for t in LATE_TRIPS:
        ran[t] = shift_trip(ran[t], LATE_SECONDS)
    for t in EARLY_ARRIVAL_TRIPS:
        ran[t] = shift_arrival(ran[t], LAST_TIMEPOINT, -EARLY_ARRIVAL_SECONDS)

    return Scenario(
        name="mixed_punctuality",
        trips=ran,
        scheduled_trips=scheduled,
        notes=[
            "10 trips every 15 minutes from 04:15, all of them Early AM",
            "4 run exactly to time, at every timepoint",
            "3 run 360 s late at every timepoint",
            "3 reach the last timepoint 120 s early, and are on time before it",
            "stops 1 and 2: 7 deviations of 0 s and 3 of +360 s",
            "stop 5: 4 of 0 s, 3 of -120 s and 3 of +360 s",
        ],
    )


def punctuality_thresholds() -> Scenario:
    """Trips landing exactly on the on time thresholds, and one second past them.

    On time performance turns a continuous deviation into three buckets, so the
    only interesting question about the boundary is which side of it the boundary
    itself falls on. "A minute early or more" and "five minutes late or more"
    are both inclusive, so a trip exactly 60 s early and one exactly 300 s late
    are **not on time**, and neither are the trips a second beyond them.

    Eight trips: one exactly on each threshold, one a second past each, and four
    dead on time. Every stop time of a trip moves together, so each trip carries
    one classification the whole way down the line and every timepoint reports
    the same split -- what changes between the two punctuality scenarios is that
    this one puts the interest in the thresholds and none of it in the stops.

    The deviations are spread far enough apart to make both reported percentiles
    land inside an interpolation rather than on a repeated zero, which
    mixed_punctuality's mostly-on-time sample cannot do.

    Trips run every 15 minutes from 04:30, so all of them fall inside Early AM.
    """
    starts = even_starts(
        midnight(DATE) + datetime.timedelta(hours=4, minutes=30), 15, 8
    )
    scheduled = [
        build_trip(f"boundary-{t:02d}", DATE, start) for t, start in enumerate(starts)
    ]
    ran = [
        shift_trip(trip, BOUNDARY_DEVIATIONS.get(t, 0))
        for t, trip in enumerate(scheduled)
    ]

    return Scenario(
        name="punctuality_thresholds",
        trips=ran,
        scheduled_trips=scheduled,
        notes=[
            "8 trips every 15 minutes from 04:30, all of them Early AM",
            "deviations at every timepoint: 0, -60, -61, +300, +301, 0, 0, 0",
            "-60 and +300 sit exactly on the thresholds, so neither is on time",
            "-60 and -61 are early, +300 and +301 are late",
            "a whole trip moves together, so travel times are untouched",
        ],
    )


SCENARIOS = [
    constant_headway,
    variable_headway,
    variable_travel_time,
    two_service_dates,
    inconsistent_feed,
    mixed_punctuality,
    punctuality_thresholds,
]


# ---------------------------------------------------------------------------
# Building the tables
# ---------------------------------------------------------------------------


def build_schedule(scenario: Scenario) -> list[Trip]:
    """The trips as dim_stop_arrivals publishes them, rather than as they ran.

    By default every scenario's trips are laid out on the same even 15 minute
    grid from 06:00, whatever the scenario actually ran. That is exactly what
    constant_headway runs, so there schedule and realtime agree trip for trip;
    where a scenario runs something else against the same even schedule, the
    difference between the two is what makes an excess wait measurable at all.

    A scenario carrying its own ``scheduled_trips`` publishes those instead --
    either because where the seconds are counted from is its whole point, or
    because it needs a timetable its trips can then be measured late against.
    """
    if scenario.scheduled_trips is not None:
        return distinct_trips(scenario.scheduled_trips)

    # one published trip per trip_id, however many dates it ran on
    published = distinct_trips(scenario.trips)

    # the same trip_ids, so a realtime stop time still finds its scheduled
    # counterpart -- that join is on trip_id, stop_id and stop_sequence
    grid = even_starts(midnight(DATE) + datetime.timedelta(hours=6), 15, len(published))
    return [
        build_trip(trip.trip_id, DATE, start) for trip, start in zip(published, grid)
    ]


def build_tables(scenario: Scenario) -> dict[str, list[dict]]:
    trips = scenario.trips
    service_dates = sorted({trip.service_date for trip in trips})

    return {
        "fct_scheduled_trips": [
            {
                "service_date": trip.service_date,
                "feed_key": FEED_KEY,
                "trip_id": trip.trip_id,
                "shape_id": SHAPE_ID,
                "direction_id": 0,
                "route_id": ROUTE_ID,
                "route_short_name": ROUTE_SHORT_NAME,
            }
            for trip in trips
        ],
        "fct_stop_time_metrics": [
            {
                "service_date": trip.service_date,
                "schedule_base64_url": SCHEDULE_BASE64_URL,
                "trip_id": trip.trip_id,
                "trip_key": trip.trip_key,
                "stop_id": stop_id,
                "stop_sequence": sequence,
                "actual_arrival_pacific": arrival,
                "actual_departure_pacific": departure,
                # carried through to the metric input but read by neither metric
                "n_predictions": 0,
            }
            for trip in trips
            for sequence, stop_id, arrival, departure in trip.stop_times
        ],
        "dim_stop_times": [
            {
                "feed_key": FEED_KEY,
                "trip_id": trip.trip_id,
                "stop_id": stop_id,
                "stop_sequence": sequence,
                "timepoint": 1 if sequence in TIMEPOINT_SEQUENCES else 0,
                "_feed_valid_from": FEED_VALID_FROM,
            }
            # grained by feed version rather than service date, as dim_stop_arrivals is
            for trip in distinct_trips(trips)
            for sequence, stop_id, _, _ in trip.stop_times
        ],
        # grained by feed version rather than service date, so it carries
        # neither -- what ties a published time to a realtime one is the trip_id
        "dim_stop_arrivals": [
            {
                "feed_key": FEED_KEY,
                "trip_id": trip.trip_id,
                "stop_id": stop_id,
                "stop_sequence": sequence,
                "arrival_sec": seconds_since_midnight(arrival, trip.service_date),
                "departure_sec": seconds_since_midnight(departure, trip.service_date),
                "_feed_valid_from": FEED_VALID_FROM,
            }
            for trip in build_schedule(scenario)
            for sequence, stop_id, arrival, departure in trip.stop_times
        ],
        # stops repeat once per service date, and the templates collapse them
        "fct_daily_scheduled_stops": [
            {
                "service_date": date,
                "feed_key": FEED_KEY,
                "stop_id": stop_id,
                "pt_geom": stop_point(sequence),
            }
            for date in service_dates
            for sequence, stop_id in enumerate(STOP_IDS, start=1)
        ],
        "dim_schedule_feeds": [
            {
                "key": FEED_KEY,
                "base64_url": SCHEDULE_BASE64_URL,
                "_valid_from": FEED_VALID_FROM,
            }
        ],
        "fct_daily_schedule_feeds": [
            {
                "date": midnight(date),
                "feed_key": FEED_KEY,
                "gtfs_dataset_name": GTFS_DATASET_NAME,
            }
            for date in service_dates
        ],
        # two points is all make_linestring needs to build a real LineString
        "fct_daily_scheduled_shapes": [
            {
                "service_date": date,
                "feed_key": FEED_KEY,
                "shape_id": SHAPE_ID,
                "pt_array": [stop_point(1), stop_point(len(STOP_IDS))],
            }
            for date in service_dates
        ],
    }


def write(out_dir: pathlib.Path, table: str, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=TABLES[table]), out_dir / f"{table}.parquet"
    )
    print(f"  {table}: {len(rows)} rows")


if __name__ == "__main__":
    for build in SCENARIOS:
        scenario = build()
        print(f"{scenario.name}/")
        for note in scenario.notes:
            print(f"  # {note}")
        for table, rows in build_tables(scenario).items():
            write(OUT_ROOT / scenario.name, table, rows)
