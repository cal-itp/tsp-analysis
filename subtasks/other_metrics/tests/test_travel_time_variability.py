"""Trip duration variability, played end to end over the cassettes.

Each test names a cassette and says what the reported table should be. Nothing
in between is asserted: the loaders are private, so the run goes from parquet
straight to ``calculate_travel_time_variability``'s output.
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

N_TRIPS = {
    "constant_headway": 8,
    "variable_headway": 24,
    "variable_travel_time": 9,
    "two_service_dates": 8,
    "inconsistent_feed": 8,
    "mixed_punctuality": 10,
    "punctuality_thresholds": 8,
}

# the cassettes covering more than one service date. the segment grain has no
# service_date, so both of them pool their dates into a single sample
MULTI_DATE_CASSETTES = ["two_service_dates", "inconsistent_feed"]

# the cassettes where every trip runs the line in the same time. in
# two_service_dates that holds across both of its dates, which the segment grain
# pools together. punctuality_thresholds runs its trips early and late, but moves
# each one's stop times together -- a late bus is late, not slow -- so its travel
# times are untouched and it belongs here too
CONSTANT_TRAVEL_TIME_CASSETTES = [
    "constant_headway",
    "variable_headway",
    "two_service_dates",
    "punctuality_thresholds",
]

# in variable_travel_time every third trip runs the whole line in half the time
FAST_SEGMENT_FACTOR = 0.5

# the stop inconsistent_feed contradicts itself about: one trip on each date
# reports reaching it before it left the stop before
DISORDERED_SEQUENCE = 2

EXPECTED_COLUMNS = [
    "shape_id",
    "feed_key",
    "stop_sequence",
    "travel_time_variability",
    "p80_travel_time_seconds",
    "p20_travel_time_seconds",
    "n_rt_trips",
    "n_service_dates",
]


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

    def test_index_is_reset(self, travel_time_variability):
        # the grain comes back as columns rather than a MultiIndex, so the
        # notebook can merge the geometry straight onto it
        assert travel_time_variability.index.to_list() == list(
            range(len(travel_time_variability))
        )

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_one_row_per_timepoint_segment(self, travel_time_variability):
        """Timepoints only, one row each -- the segment grain is the reporting grain."""
        assert travel_time_variability["stop_sequence"].to_list() == TIMEPOINT_SEQUENCES

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
        assert set(travel_time_variability["n_rt_trips"]) == {N_TRIPS[cassette_name]}

    @pytest.mark.parametrize(
        "cassette_name",
        [name for name in CASSETTES if name not in MULTI_DATE_CASSETTES],
    )
    def test_single_date_cassettes_report_one_service_date(
        self, travel_time_variability
    ):
        assert set(travel_time_variability["n_service_dates"]) == {1}

    @pytest.mark.parametrize("cassette_name", MULTI_DATE_CASSETTES)
    def test_service_dates_are_pooled(self, travel_time_variability):
        """The segment grain has no service_date, so both dates land on one row.

        4 trips on each date, pooled into a single sample of 8 per segment
        rather than reported a date at a time.
        """
        assert set(travel_time_variability["n_service_dates"]) == {2}
        assert set(travel_time_variability["n_rt_trips"]) == {8}

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_a_trip_id_running_on_two_dates_is_two_trips(self, travel_time_variability):
        """trip_id is only unique within a service date, so the date is in the grain.

        inconsistent_feed runs the same four trip ids on both of its dates. That
        is 8 trips, not 4, and no travel time may be measured from one date's
        stop to the other's -- a grain that had dropped service_date would pair
        6/17's stops with 6/18's and report a travel time spanning the night.
        """
        assert set(travel_time_variability["n_rt_trips"]) == {8}

        measured = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] != FIRST_TIMEPOINT
        ]
        assert measured["p80_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]


class TestValues:
    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_first_timepoint_has_no_travel_time(self, travel_time_variability):
        """The docstring promises a null ratio where there is no preceding stop."""
        first = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] == FIRST_TIMEPOINT
        ]

        assert len(first) == 1
        assert first["travel_time_variability"].isna().all()
        assert first["p80_travel_time_seconds"].isna().all()
        assert first["p20_travel_time_seconds"].isna().all()
        # the trip is still counted, only its travel time is undefined
        assert (first["n_rt_trips"] > 0).all()

    @pytest.mark.parametrize("cassette_name", CONSTANT_TRAVEL_TIME_CASSETTES)
    def test_constant_travel_time_is_one(self, travel_time_variability):
        """Every trip runs the line in the same time, so p80 and p20 agree.

        two_service_dates runs the same pattern either side of a date boundary
        and over GTFS hours past 24, and must not budge from it.
        """
        measured = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] != FIRST_TIMEPOINT
        ]

        assert measured["stop_sequence"].to_list() == MEASURED_SEQUENCES
        assert (measured["travel_time_variability"] == 1.0).all()
        assert measured["p80_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]
        assert measured["p20_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]

    @pytest.mark.parametrize("cassette_name", ["variable_travel_time"])
    def test_variable_travel_time_is_two(self, travel_time_variability):
        """A third of the trips run the line in half the time, so the ratio is 2.

        Travel time takes exactly two values on each segment, and a third of the
        trips carry the fast one, so p20 lands inside the fast group and p80
        inside the slow one with a whole trip of room either side. The ratio is
        the ratio of the two values themselves, not an interpolation between them.
        """
        measured = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] != FIRST_TIMEPOINT
        ]

        assert measured["stop_sequence"].to_list() == MEASURED_SEQUENCES
        assert (measured["travel_time_variability"] == 2.0).all()
        assert measured["p80_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]
        assert measured["p20_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] * FAST_SEGMENT_FACTOR
            for sequence in MEASURED_SEQUENCES
        ]

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
        segment = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] == DISORDERED_SEQUENCE
        ].iloc[0]

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
        segment = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] == 5
        ].iloc[0]

        assert segment["p80_travel_time_seconds"] == SEGMENT_SECONDS[5]
        assert segment["p20_travel_time_seconds"] == SEGMENT_SECONDS[5]
        assert segment["travel_time_variability"] == 1.0

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_the_ratio_is_the_two_percentiles(self, travel_time_variability):
        """Stated as an identity, so a cassette added later still checks it."""
        measured = travel_time_variability.loc[
            travel_time_variability["stop_sequence"] != FIRST_TIMEPOINT
        ]
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
        """p90/p10 still brackets the same two travel times, so the ratio holds.

        variable_travel_time has 3 fast trips and 6 slow ones, so p10 is inside
        the fast group and p90 inside the slow one just as p20 and p80 are. Only
        a percentile crossing into the other group would move the ratio.
        """
        result = run_travel_time_variability(
            "variable_travel_time", upper_percentile=0.9, lower_percentile=0.1
        )
        measured = result.loc[result["stop_sequence"] != FIRST_TIMEPOINT]

        assert (measured["travel_time_variability"] == 2.0).all()
        assert measured["p90_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] for sequence in MEASURED_SEQUENCES
        ]
        assert measured["p10_travel_time_seconds"].to_list() == [
            SEGMENT_SECONDS[sequence] * FAST_SEGMENT_FACTOR
            for sequence in MEASURED_SEQUENCES
        ]
