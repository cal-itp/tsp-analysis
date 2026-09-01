"""Trip duration variability, played end to end over the cassettes.

Each test names a cassette and says what the reported table should be. Nothing
in between is asserted: the loaders are private, so the run goes from parquet
straight to ``calculate_travel_time_variability``'s output.

The reporting grain is the segment and the day part. A run is placed by the
clock hour of the actual arrival that ends it -- realtime throughout, unlike on
time performance's scheduled hour -- and trips from every selected service date
are pooled inside each day part.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import CASSETTES, run_travel_time_variability

# stops 1, 2 and 5 are timepoints on every trip, so the metric reports three
# segments. Stop 1 leads the pattern and has no preceding stop, which leaves two
# measured segments of deliberately different length
TIMEPOINT_SEQUENCES = [1, 2, 5]
FIRST_TIMEPOINT = 1
MEASURED_SEQUENCES = [2, 5]

# 120 s of running into stop 2; 120 s three times over plus two 30 s dwells into
# stop 5. Every cassette schedules these, and all but variable_travel_time run them
SEGMENT_SECONDS = {2: 120.0, 5: 420.0}

# the stops the headway metric reports on but this one does not: not timepoints,
# so they carry no timepoint-to-timepoint travel time
NON_TIMEPOINT_SEQUENCES = [3, 4]

# the day parts, in the order the clock runs through them from midnight. Stated
# here independently of the implementation, and the order the metric reports in:
# down the line, then round the clock. two_service_dates is the one that shows
# the difference -- its service runs Evening, Owl, Early AM, and it is reported
# Owl, Early AM, Evening
PERIOD_NAMES = ["Owl", "Early AM", "AM Peak", "Midday", "PM Peak", "Evening"]

# how the trips at one timepoint fall across the day parts, cassette by cassette.
# Every timepoint sees the same split: a trip runs the whole line in ten minutes
# and no cassette starts one close enough to a boundary to cross it mid-run, so a
# trip's three timepoints are always in the one day part
TRIPS_PER_TIMEPOINT = {
    # 06:00 to 07:45 every 15 minutes, so four trips land either side of 07:00
    "constant_headway": {"Early AM": 4, "AM Peak": 4},
    # two blocks by construction: alternating gaps 07:00 to 09:40, an even 15
    # from 10:00
    "variable_headway": {"AM Peak": 12, "Midday": 12},
    # 06:00 to 08:00 every 15 minutes, so the nine trips split 4 and 5
    "variable_travel_time": {"Early AM": 4, "AM Peak": 5},
    # 6/17's four trips start at 23:30 and run past midnight; 6/18's four start
    # at 04:00. The date boundary and the day part boundary are not the same line
    "two_service_dates": {"Evening": 2, "Owl": 2, "Early AM": 4},
    # four trips from 06:00 on each of two dates, pooled into one day part
    "inconsistent_feed": {"Early AM": 8},
    # 04:15 to 06:30, and the six minutes the late trips run down does not reach
    # 07:00
    "mixed_punctuality": {"Early AM": 10},
    # 04:30 to 06:15, and the five minutes the latest trip runs down does not
    # reach 07:00
    "punctuality_thresholds": {"Early AM": 8},
}

# how many trips each cassette runs in total, stated separately from the split
# above so that the two have to agree
N_TRIPS = {
    "constant_headway": 8,
    "variable_headway": 24,
    "variable_travel_time": 9,
    "two_service_dates": 8,
    "inconsistent_feed": 8,
    "mixed_punctuality": 10,
    "punctuality_thresholds": 8,
}

# the cassettes covering more than one service date, and what the day part grain
# does to them. inconsistent_feed runs both dates through the same day part, so
# its dates pool onto one row. two_service_dates runs each of its dates through
# day parts of its own, so every row sees a single date -- a day part is a clock
# hour, while a service date is what a trip is listed under
POOLED_DATE_CASSETTES = ["inconsistent_feed"]
SPLIT_DATE_CASSETTES = ["two_service_dates"]

# the cassettes where every trip runs the line in the same time. in
# two_service_dates that holds across both of its dates and all three of its day
# parts. punctuality_thresholds runs its trips early and late, but moves each
# one's stop times together -- a late bus is late, not slow -- so its travel
# times are untouched and it belongs here too
CONSTANT_TRAVEL_TIME_CASSETTES = [
    "constant_headway",
    "variable_headway",
    "two_service_dates",
    "punctuality_thresholds",
]

# in variable_travel_time every third trip runs the whole line in half the time
FAST_SEGMENT_FACTOR = 0.5

# and the day part decides how much of that reaches the ratio. Early AM holds
# trips 1 to 4, two of them fast; AM Peak holds trips 5 to 9, one of them fast
FAST_PERIOD = "Early AM"
SLOW_PERIOD = "AM Peak"

# Early AM: four values, two fast and two slow. p20 lands at 0.2 * (4 - 1) = 0.6,
# inside the fast pair, and p80 at 2.4, inside the slow pair -- so the ratio is
# the ratio of the two travel times themselves, 2.0, as it is over the whole day
FAST_PERIOD_VARIABILITY = 1 / FAST_SEGMENT_FACTOR

# AM Peak: five values, one fast and four slow. p80 is inside the slow group at
# index 3.2, but p20 lands at 0.8 -- between the single fast value and the first
# slow one, so it interpolates 80% of the way from one to the other rather than
# landing inside either group:
#     p20 = fast + 0.8 * (slow - fast)
# which for a fast time of half the slow one is 0.9 of the slow. The ratio is
# therefore 1 / 0.9 whatever the segment's length
SLOW_PERIOD_P20_FACTOR = FAST_SEGMENT_FACTOR + 0.8 * (1 - FAST_SEGMENT_FACTOR)
SLOW_PERIOD_VARIABILITY = 1 / SLOW_PERIOD_P20_FACTOR

# the stop inconsistent_feed contradicts itself about: one trip on each date
# reports reaching it before it left the stop before
DISORDERED_SEQUENCE = 2

# mixed_punctuality's three early trips are early at the last timepoint only, so
# they reach stop 5 two minutes sooner without leaving stop 2 any later. That is
# a shorter run, not a late one, and it is the only cassette outside
# variable_travel_time whose travel time moves at all
EARLY_ARRIVAL_SECONDS = 120.0

EXPECTED_COLUMNS = [
    "shape_id",
    "feed_key",
    "stop_sequence",
    "time_period",
    "travel_time_variability",
    "p80_travel_time_seconds",
    "p20_travel_time_seconds",
    "n_rt_trips",
    "n_service_dates",
]


def rows_for(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    for column, value in filters.items():
        frame = frame.loc[frame[column] == value]
    return frame


def one_row(frame: pd.DataFrame, **filters) -> pd.Series:
    matching = rows_for(frame, **filters)
    assert len(matching) == 1, f"{filters} matched {len(matching)} rows"
    return matching.iloc[0]


def measured_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Everything but the leading timepoint, which has no preceding stop."""
    return frame.loc[frame["stop_sequence"] != FIRST_TIMEPOINT]


class TestOutputFormat:
    def test_schema(self, travel_time_variability):
        assert list(travel_time_variability.columns) == EXPECTED_COLUMNS

    def test_dtypes(self, travel_time_variability):
        assert travel_time_variability["shape_id"].dtype == object
        assert travel_time_variability["feed_key"].dtype == object
        assert travel_time_variability["stop_sequence"].dtype.kind == "i"
        assert travel_time_variability["travel_time_variability"].dtype.kind == "f"
        assert travel_time_variability["p80_travel_time_seconds"].dtype.kind == "f"
        assert travel_time_variability["p20_travel_time_seconds"].dtype.kind == "f"
        assert travel_time_variability["n_rt_trips"].dtype.kind == "i"
        assert travel_time_variability["n_service_dates"].dtype.kind == "i"

    def test_the_day_part_is_an_ordered_categorical(self, travel_time_variability):
        """Ordered, so the table sorts by the clock rather than the alphabet."""
        period = travel_time_variability["time_period"]

        assert isinstance(period.dtype, pd.CategoricalDtype)
        assert period.cat.ordered
        assert list(period.cat.categories) == PERIOD_NAMES

    def test_index_is_reset(self, travel_time_variability):
        # the grain comes back as columns rather than a MultiIndex, so the
        # notebook can merge the geometry straight onto it
        assert travel_time_variability.index.to_list() == list(
            range(len(travel_time_variability))
        )

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_one_row_per_timepoint_segment_and_day_part(
        self, travel_time_variability, cassette_name
    ):
        """Every timepoint and period that saw service gets a row, and nothing else.

        A period with no service on a segment must not appear at all -- grouping
        on a categorical will happily invent those rows, and they would carry a
        zero count and a NaN ratio indistinguishable from the leading timepoint's.
        """
        periods = TRIPS_PER_TIMEPOINT[cassette_name]
        expected = [
            (sequence, period)
            for sequence in TIMEPOINT_SEQUENCES
            for period in PERIOD_NAMES
            if period in periods
        ]

        reported = travel_time_variability[["stop_sequence", "time_period"]]

        assert list(reported.itertuples(index=False, name=None)) == expected

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_stops_that_are_not_timepoints_are_absent(self, travel_time_variability):
        """A stop flagged as a timepoint on only some trips is not one."""
        reported = set(travel_time_variability["stop_sequence"])

        assert reported.isdisjoint(NON_TIMEPOINT_SEQUENCES)

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_one_shape_and_feed(self, travel_time_variability):
        assert set(travel_time_variability["shape_id"]) == {"shape1"}
        assert set(travel_time_variability["feed_key"]) == {"feed1"}


class TestCounts:
    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_every_trip_is_counted_on_every_segment(
        self, travel_time_variability, cassette_name
    ):
        """Each trip is counted once per timepoint, in the day part it ran in.

        The counts are per day part now rather than per segment, so the check
        that nothing was lost or double counted is that they add back up to the
        cassette's trips at every timepoint.
        """
        periods = TRIPS_PER_TIMEPOINT[cassette_name]

        for sequence in TIMEPOINT_SEQUENCES:
            at_timepoint = rows_for(travel_time_variability, stop_sequence=sequence)
            counts = dict(
                zip(
                    at_timepoint["time_period"], at_timepoint["n_rt_trips"], strict=True
                )
            )

            assert counts == periods, sequence
            assert sum(counts.values()) == N_TRIPS[cassette_name], sequence

    @pytest.mark.parametrize(
        "cassette_name",
        [name for name in CASSETTES if name not in POOLED_DATE_CASSETTES],
    )
    def test_a_day_part_fed_by_one_date_reports_one_service_date(
        self, travel_time_variability
    ):
        """Only inconsistent_feed runs two dates through the same day part.

        two_service_dates covers two dates as well, but its day parts each see
        only one of them, so it belongs here rather than with the pooling.
        """
        assert set(travel_time_variability["n_service_dates"]) == {1}

    @pytest.mark.parametrize("cassette_name", POOLED_DATE_CASSETTES)
    def test_dates_are_pooled_inside_a_day_part(self, travel_time_variability):
        """The grain has no service_date, so both dates land on one row.

        inconsistent_feed runs 4 trips from 06:00 on each of its dates, so all 8
        are Early AM and pool into a single sample per segment rather than being
        reported a date at a time.
        """
        assert set(travel_time_variability["n_service_dates"]) == {2}
        assert set(travel_time_variability["n_rt_trips"]) == {8}

    @pytest.mark.parametrize("cassette_name", SPLIT_DATE_CASSETTES)
    def test_a_day_part_is_a_clock_hour_not_a_service_date(
        self, travel_time_variability
    ):
        """two_service_dates' dates land in day parts of their own.

        6/17's service runs 23:30 to 00:25 and 6/18's from 04:00, so Evening and
        Owl are fed by 6/17 alone and Early AM by 6/18 alone. Every row therefore
        reports a single service date even though the window covers two -- and
        Owl holds 6/17's trips, not 6/18's, because a trip past midnight is still
        listed under the date it set out on.
        """
        assert set(travel_time_variability["n_service_dates"]) == {1}
        assert set(travel_time_variability["time_period"]) == {
            "Evening",
            "Owl",
            "Early AM",
        }

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_a_trip_id_running_on_two_dates_is_two_trips(self, travel_time_variability):
        """trip_id is only unique within a service date, so the date is in the grain.

        inconsistent_feed runs the same four trip ids on both of its dates. That
        is 8 trips, not 4, and no travel time may be measured from one date's
        stop to the other's -- a grain that had dropped service_date would pair
        6/17's stops with 6/18's and report a travel time spanning the night.
        Both dates run the same hours, so the day part does not separate them
        either.
        """
        assert set(travel_time_variability["n_rt_trips"]) == {8}

        measured = measured_rows(travel_time_variability)
        assert measured["p80_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]


class TestValues:
    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_first_timepoint_has_no_travel_time(
        self, travel_time_variability, cassette_name
    ):
        """The docstring promises a null ratio where there is no preceding stop."""
        first = rows_for(travel_time_variability, stop_sequence=FIRST_TIMEPOINT)

        assert len(first) == len(TRIPS_PER_TIMEPOINT[cassette_name])
        assert first["travel_time_variability"].isna().all()
        assert first["p80_travel_time_seconds"].isna().all()
        assert first["p20_travel_time_seconds"].isna().all()
        # the trips are still counted, only their travel time is undefined
        assert (first["n_rt_trips"] > 0).all()

    @pytest.mark.parametrize("cassette_name", CONSTANT_TRAVEL_TIME_CASSETTES)
    def test_constant_travel_time_is_one(self, travel_time_variability, cassette_name):
        """Every trip runs the line in the same time, so p80 and p20 agree.

        two_service_dates runs the same pattern either side of a date boundary
        and over GTFS hours past 24, and must not budge from it in any of the
        three day parts it lands in.
        """
        measured = measured_rows(travel_time_variability)
        periods = TRIPS_PER_TIMEPOINT[cassette_name]

        assert len(measured) == len(MEASURED_SEQUENCES) * len(periods)
        assert (measured["travel_time_variability"] == 1.0).all()
        for sequence in MEASURED_SEQUENCES:
            at_timepoint = rows_for(measured, stop_sequence=sequence)
            for column in ["p80_travel_time_seconds", "p20_travel_time_seconds"]:
                assert set(at_timepoint[column]) == {SEGMENT_SECONDS[sequence]}

    @pytest.mark.parametrize("cassette_name", ["variable_travel_time"])
    def test_a_day_part_holding_a_third_of_the_fast_trips_reports_two(
        self, travel_time_variability
    ):
        """Early AM holds two fast trips out of four, so the ratio is 2.0.

        A third of the trips carry the fast travel time over the day as a whole,
        and Early AM is the half of it that keeps the proportion: p20 lands
        inside the fast pair and p80 inside the slow one, so the ratio is the
        ratio of the two travel times themselves rather than an interpolation
        between them.
        """
        for sequence in MEASURED_SEQUENCES:
            row = one_row(
                travel_time_variability,
                stop_sequence=sequence,
                time_period=FAST_PERIOD,
            )

            assert row["travel_time_variability"] == FAST_PERIOD_VARIABILITY
            assert row["p80_travel_time_seconds"] == SEGMENT_SECONDS[sequence]
            assert (
                row["p20_travel_time_seconds"]
                == SEGMENT_SECONDS[sequence] * FAST_SEGMENT_FACTOR
            )

    @pytest.mark.parametrize("cassette_name", ["variable_travel_time"])
    def test_a_day_part_holding_one_fast_trip_interpolates_across_the_gap(
        self, travel_time_variability
    ):
        """AM Peak holds one fast trip out of five, which p20 does not land inside.

        The day part grain changes what the percentiles see. One fast value in
        five puts p20 at index 0.8, between the fast value and the first slow
        one, so it comes back four fifths of the way to the slow group and the
        ratio collapses from 2.0 to 1/0.9 -- the same service, split at 07:00.

        This is the case that says the split is real. A run that had kept
        pooling the whole day would report 2.0 here.
        """
        for sequence in MEASURED_SEQUENCES:
            row = one_row(
                travel_time_variability,
                stop_sequence=sequence,
                time_period=SLOW_PERIOD,
            )

            assert row["travel_time_variability"] == pytest.approx(
                SLOW_PERIOD_VARIABILITY
            )
            assert row["p80_travel_time_seconds"] == SEGMENT_SECONDS[sequence]
            assert row["p20_travel_time_seconds"] == pytest.approx(
                SEGMENT_SECONDS[sequence] * SLOW_PERIOD_P20_FACTOR
            )

    @pytest.mark.parametrize("cassette_name", ["mixed_punctuality"])
    def test_arriving_early_at_one_stop_is_a_shorter_run_into_it(
        self, travel_time_variability
    ):
        """Three of the ten trips reach stop 5 two minutes early, having left stop 2 on time.

        Nothing about this trip is late or slow, so the metric has no view on it
        at all until the last segment -- where the same run took 300 s rather
        than 420 s. Sorted, that is three 300s and seven 420s over ten trips, so
        p20 lands at index 1.8, inside the 300s, and p80 at 7.2, inside the 420s.

        All ten trips run 04:15 to 06:40, so the whole cassette is one day part
        and this is the pooled answer as well as the Early AM one.
        """
        row = one_row(travel_time_variability, stop_sequence=5)
        short_run = SEGMENT_SECONDS[5] - EARLY_ARRIVAL_SECONDS

        assert row["n_rt_trips"] == N_TRIPS["mixed_punctuality"]
        assert row["p80_travel_time_seconds"] == SEGMENT_SECONDS[5]
        assert row["p20_travel_time_seconds"] == short_run
        assert row["travel_time_variability"] == SEGMENT_SECONDS[5] / short_run

        # the run into stop 2 ends where it always did, so it is untouched
        into_stop_2 = one_row(travel_time_variability, stop_sequence=2)
        assert into_stop_2["travel_time_variability"] == 1.0

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_a_negative_travel_time_is_dropped(self, travel_time_variability):
        """A stop reached before the bus left the one before it is not a travel time.

        One trip on each date reports arriving at stop 2 twenty seconds before it
        departed stop 1, so 2 of that segment's 8 observations are negative. They
        are discarded rather than let into the percentiles: the six real ones are
        all 120 s, so both percentiles are 120 s and the ratio is 1.0.

        Keeping them would put p20 below the fastest trip that actually ran --
        two negatives out of eight puts the 20th percentile inside them -- and
        report a ratio under 1 for a segment every real trip ran in the same time.
        """
        segment = one_row(travel_time_variability, stop_sequence=DISORDERED_SEQUENCE)

        assert segment["p20_travel_time_seconds"] == SEGMENT_SECONDS[2]
        assert segment["p80_travel_time_seconds"] == SEGMENT_SECONDS[2]
        assert segment["travel_time_variability"] == 1.0

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_a_trip_with_no_usable_travel_time_is_still_counted(
        self, travel_time_variability
    ):
        """Only the travel time is discarded, not the trip.

        n_rt_trips is how many trips ran the segment, not how many contributed a
        travel time, so all 8 are counted on stop 2 even though 6 fed the
        percentiles. Stop 5 is untouched by the disorder and runs all 8 to time,
        so the two segments report the same count from different footing.

        The bad arrival is 20 s before the departure it follows, which is far too
        small to move the trip out of Early AM -- a stop time can be wrong enough
        to lose its travel time and still be placed on the clock.
        """
        counts = travel_time_variability.set_index("stop_sequence")["n_rt_trips"]

        assert counts[DISORDERED_SEQUENCE] == N_TRIPS["inconsistent_feed"]
        assert counts[5] == N_TRIPS["inconsistent_feed"]

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_the_disorder_does_not_reach_the_following_segment(
        self, travel_time_variability
    ):
        """The bad arrival is at stop 2, and the segment into stop 5 starts there.

        Only the arrival moved -- the departure from stop 2 is where it always
        was -- so a trip that contradicted itself on the way in still runs the
        rest of the line to time, and all 8 trips report the usual 420 s.
        """
        segment = one_row(travel_time_variability, stop_sequence=5)

        assert segment["p80_travel_time_seconds"] == SEGMENT_SECONDS[5]
        assert segment["p20_travel_time_seconds"] == SEGMENT_SECONDS[5]
        assert segment["travel_time_variability"] == 1.0

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_the_ratio_is_the_two_percentiles(self, travel_time_variability):
        """Stated as an identity, so a cassette added later still checks it."""
        measured = measured_rows(travel_time_variability)
        expected = (
            measured["p80_travel_time_seconds"] / measured["p20_travel_time_seconds"]
        )

        pd.testing.assert_series_equal(
            measured["travel_time_variability"], expected, check_names=False
        )


class TestPercentileArguments:
    def test_the_percentiles_name_the_columns(self):
        """The reported column names follow the percentiles asked for."""
        result = run_travel_time_variability(
            "variable_travel_time", upper_percentile=0.9, lower_percentile=0.1
        )

        assert "p90_travel_time_seconds" in result.columns
        assert "p10_travel_time_seconds" in result.columns
        assert "p80_travel_time_seconds" not in result.columns

    def test_widening_the_percentiles_does_not_change_a_two_valued_sample(self):
        """p90/p10 still brackets the same two travel times in Early AM.

        Early AM holds two fast trips and two slow ones, so p10 is inside the
        fast pair and p90 inside the slow one just as p20 and p80 are. Only a
        percentile crossing into the other group would move the ratio.
        """
        result = run_travel_time_variability(
            "variable_travel_time", upper_percentile=0.9, lower_percentile=0.1
        )
        early = rows_for(measured_rows(result), time_period=FAST_PERIOD)

        assert early["stop_sequence"].to_list() == MEASURED_SEQUENCES
        assert (early["travel_time_variability"] == FAST_PERIOD_VARIABILITY).all()
        assert early["p90_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]
        assert early["p10_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] * FAST_SEGMENT_FACTOR
            for sequence in MEASURED_SEQUENCES
        ]
