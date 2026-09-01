"""On time performance, played end to end over the cassettes.

Each test names a cassette and says what the two reported tables should be.
Nothing in between is asserted: the loaders are private, so the run goes from
parquet straight to ``calculate_on_time_performance``'s output.

What the metric does, restated here so a change to it has to be made in two
places:

* The deviation is the actual arrival at a timepoint less the arrival the
  schedule published for that trip at that stop, in seconds, signed so that late
  is positive.
* A vehicle is **early** if it beats its published arrival by ``early_seconds``
  or more, **late** if it misses it by ``late_seconds`` or more, and **not on
  time** if it is either. All three are proportions of the timepoint arrivals in
  the group, and the two thresholds are inclusive.
* ``schedule_adherence`` is one less ``proportion_not_on_time``: the share of
  arrivals that made their published time.
* The day part comes from the **scheduled** arrival, not the actual one, so a
  trip running late is still reported in the day part it was promised in.

None of the cassettes have any jitter, so every value below is exact and derived
from the service the fixture describes rather than read back off a previous run.
Where a rounded literal appears it carries its arithmetic in the docstring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import CASSETTES, run_on_time_performance

# stops 1, 2 and 5 are timepoints on every trip. On time performance reports
# every one of them -- unlike travel time variability, the first has no
# predecessor to need, so nothing is undefined there
TIMEPOINT_SEQUENCES = [1, 2, 5]
LAST_TIMEPOINT = 5
BEFORE_LAST_TIMEPOINT = [1, 2]
NON_TIMEPOINT_SEQUENCES = [3, 4]

STOP_IDS = {1: "stop1", 2: "stop2", 5: "stop5"}

# the thresholds, stated here independently of the implementation so moving one
# has to be done in both places. Both are inclusive: a vehicle exactly this early
# or exactly this late is not on time
EARLY_THRESHOLD_SECONDS = 60
LATE_THRESHOLD_SECONDS = 300

# the cassettes whose vehicles hit every published arrival exactly. Everything
# these report is zero, which is the only way to tell a metric that measures
# nothing apart from one that measures nothing wrong
PERFECT_CASSETTES = ["constant_headway", "two_service_dates"]

# ---------------------------------------------------------------------------
# mixed_punctuality: 10 trips, three 360 s late throughout, three 120 s early at
# the last timepoint only. Its whole point is that those two stops differ
# ---------------------------------------------------------------------------
MIXED_TRIPS = 10
MIXED_LATE_TRIPS = 3
MIXED_EARLY_TRIPS = 3
MIXED_LATE_SECONDS = 360
MIXED_EARLY_SECONDS = 120

# stops 1 and 2 see seven deviations of 0 s and the three late trips' +360 s. The
# sorted absolute deviations are seven zeros and three 360s, so over ten values
#     p50 = value[4] + 0.5 * (value[5] - value[4])   =   0 + 0.5 * 0   =   0.0
#     p80 = value[7] + 0.2 * (value[8] - value[7])   = 360 + 0.2 * 0   = 360.0
MIXED_BEFORE_LAST = {
    "n_timepoint_arrivals": MIXED_TRIPS,
    "p50_absolute_deviation_seconds": 0.0,
    "p80_absolute_deviation_seconds": float(MIXED_LATE_SECONDS),
    "proportion_early": 0.0,
    "proportion_late": MIXED_LATE_TRIPS / MIXED_TRIPS,
    "proportion_not_on_time": MIXED_LATE_TRIPS / MIXED_TRIPS,
    "schedule_adherence": 1 - MIXED_LATE_TRIPS / MIXED_TRIPS,
}

# the last timepoint also sees the three early trips' -120 s, which pushes the
# median off zero. Sorted absolute deviations are four zeros, three 120s and
# three 360s, so
#     p50 = value[4] + 0.5 * (value[5] - value[4])   = 120 + 0.5 * 0   = 120.0
#     p80 = value[7] + 0.2 * (value[8] - value[7])   = 360 + 0.2 * 0   = 360.0
MIXED_AT_LAST = {
    "n_timepoint_arrivals": MIXED_TRIPS,
    "p50_absolute_deviation_seconds": float(MIXED_EARLY_SECONDS),
    "p80_absolute_deviation_seconds": float(MIXED_LATE_SECONDS),
    "proportion_early": MIXED_EARLY_TRIPS / MIXED_TRIPS,
    "proportion_late": MIXED_LATE_TRIPS / MIXED_TRIPS,
    "proportion_not_on_time": (MIXED_EARLY_TRIPS + MIXED_LATE_TRIPS) / MIXED_TRIPS,
    "schedule_adherence": 1 - (MIXED_EARLY_TRIPS + MIXED_LATE_TRIPS) / MIXED_TRIPS,
}

# pooled over the three timepoints: 18 zeros, three 120s and nine 360s. p50 lands
# at the 14.5th of 30 sorted values, still inside the zeros; p80 at the 23.2nd,
# inside the 360s
MIXED_ARRIVALS = MIXED_TRIPS * len(TIMEPOINT_SEQUENCES)
MIXED_POOLED = {
    "n_timepoint_arrivals": MIXED_ARRIVALS,
    "p50_absolute_deviation_seconds": 0.0,
    "p80_absolute_deviation_seconds": float(MIXED_LATE_SECONDS),
    # the early trips are early at one timepoint each and the late ones are late
    # at all three, so neither proportion is the share of *trips* that were off schedule
    "proportion_early": MIXED_EARLY_TRIPS / MIXED_ARRIVALS,
    "proportion_late": MIXED_LATE_TRIPS * len(TIMEPOINT_SEQUENCES) / MIXED_ARRIVALS,
    "proportion_not_on_time": (
        MIXED_EARLY_TRIPS + MIXED_LATE_TRIPS * len(TIMEPOINT_SEQUENCES)
    )
    / MIXED_ARRIVALS,
    "schedule_adherence": 1
    - (MIXED_EARLY_TRIPS + MIXED_LATE_TRIPS * len(TIMEPOINT_SEQUENCES))
    / MIXED_ARRIVALS,
}

# ---------------------------------------------------------------------------
# punctuality_thresholds: 8 trips deviating by 0, -60, -61, +300, +301, 0, 0, 0
# at every timepoint. Every stop reports the same thing, and the interest is in
# which side of each threshold its two boundary trips fall
# ---------------------------------------------------------------------------

THRESHOLD_TRIPS = 8

# sorted absolute deviations are 0, 0, 0, 0, 60, 61, 300, 301, so
#     p50 = (value[3] + value[4]) / 2       = (0 + 60) / 2   =  30.0
#     p80 = value[5] + 0.6 * (value[6] - value[5]) = 61 + 0.6 * 239 = 204.4
THRESHOLD_BY_STOP = {
    "n_timepoint_arrivals": THRESHOLD_TRIPS,
    "p50_absolute_deviation_seconds": (0 + 60) / 2,
    "p80_absolute_deviation_seconds": 61 + 0.6 * (300 - 61),
    # the thresholds are inclusive, so both boundary trips count on each side
    "proportion_early": 2 / THRESHOLD_TRIPS,
    "proportion_late": 2 / THRESHOLD_TRIPS,
    "proportion_not_on_time": 4 / THRESHOLD_TRIPS,
    "schedule_adherence": 1 - 4 / THRESHOLD_TRIPS,
}

# pooled, the same eight values appear three times over: 12 zeros, three 60s,
# three 61s, three 300s and three 301s. p50 still splits a zero and a 60, but p80
# lands at the 18.4th of 24 sorted values, which is inside the 300s
THRESHOLD_ARRIVALS = THRESHOLD_TRIPS * len(TIMEPOINT_SEQUENCES)
THRESHOLD_POOLED = {
    "n_timepoint_arrivals": THRESHOLD_ARRIVALS,
    "p50_absolute_deviation_seconds": (0 + 60) / 2,
    "p80_absolute_deviation_seconds": 300.0,
    "proportion_early": 2 / THRESHOLD_TRIPS,
    "proportion_late": 2 / THRESHOLD_TRIPS,
    "proportion_not_on_time": 4 / THRESHOLD_TRIPS,
    "schedule_adherence": 1 - 4 / THRESHOLD_TRIPS,
}

# ---------------------------------------------------------------------------

# two_service_dates runs 4 trips on each of its dates, and the schedule publishes
# the second date's in GTFS hours past 24. How its timepoint arrivals fall across
# the day parts, at every timepoint
BOUNDARY_PERIODS = {"Evening": 2, "Owl": 2, "Early AM": 4}

# inconsistent_feed's bad arrival: one trip on each date reports reaching stop 2
# 20 s before it left stop 1, which is 140 s before the schedule said it would
DISORDERED_SEQUENCE = 2
DISORDERED_DEVIATION_SECONDS = -140
DISORDERED_TRIPS = 2
INCONSISTENT_TRIPS = 8

STOP_COLUMNS = [
    "shape_id",
    "stop_sequence",
    "stop_id",
    "time_period",
    "n_timepoint_arrivals",
    "p50_absolute_deviation_seconds",
    "p80_absolute_deviation_seconds",
    "proportion_early",
    "proportion_late",
    "proportion_not_on_time",
    "schedule_adherence",
]

PERIOD_COLUMNS = [
    "shape_id",
    "time_period",
    "n_timepoint_arrivals",
    "p50_absolute_deviation_seconds",
    "p80_absolute_deviation_seconds",
    "proportion_early",
    "proportion_late",
    "proportion_not_on_time",
    "schedule_adherence",
]

PROPORTION_COLUMNS = [
    "proportion_early",
    "proportion_late",
    "proportion_not_on_time",
]


@pytest.fixture
def by_stop(on_time_performance) -> pd.DataFrame:
    return on_time_performance.by_stop


@pytest.fixture
def by_period(on_time_performance) -> pd.DataFrame:
    return on_time_performance.by_period


def rows_for(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    """The rows matching every column=value pair given."""
    for column, value in filters.items():
        frame = frame.loc[frame[column] == value]
    return frame


def one_row(frame: pd.DataFrame, **filters) -> pd.Series:
    matching = rows_for(frame, **filters)
    assert len(matching) == 1, f"{filters} matched {len(matching)} rows"
    return matching.iloc[0]


def assert_reports(row: pd.Series, expected: dict) -> None:
    """Check every reported value on one row, naming the column that differed."""
    for column, value in expected.items():
        assert row[column] == pytest.approx(value), column


class TestOutputFormat:
    def test_stop_schema(self, by_stop):
        assert list(by_stop.columns) == STOP_COLUMNS

    def test_period_schema(self, by_period):
        assert list(by_period.columns) == PERIOD_COLUMNS

    def test_dtypes(self, by_stop, by_period):
        assert by_stop["shape_id"].dtype == object
        assert by_stop["stop_id"].dtype == object
        assert by_stop["stop_sequence"].dtype.kind == "i"
        for frame in [by_stop, by_period]:
            assert frame["n_timepoint_arrivals"].dtype.kind == "i"
            for column in PERIOD_COLUMNS[3:]:
                assert frame[column].dtype.kind == "f", column

    def test_time_period_is_ordered(self, by_stop, by_period):
        """Ordered, so both tables sort by the clock rather than the alphabet."""
        for frame in [by_stop, by_period]:
            assert isinstance(frame["time_period"].dtype, pd.CategoricalDtype)
            assert frame["time_period"].cat.ordered

    def test_index_is_reset(self, by_stop, by_period):
        # the grain comes back as columns rather than a MultiIndex, so the
        # notebook can merge the geometry straight onto it
        for frame in [by_stop, by_period]:
            assert frame.index.to_list() == list(range(len(frame)))


class TestTheGrain:
    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_every_reported_stop_is_a_timepoint(self, by_stop):
        """Timepoints only, the same three the travel time metric reports on."""
        reported = set(by_stop["stop_sequence"])

        assert reported == set(TIMEPOINT_SEQUENCES)
        assert reported.isdisjoint(NON_TIMEPOINT_SEQUENCES)

    # @pytest.mark.parametrize("cassette_name", CASSETTES)
    # def test_stop_id_rides_along_with_the_stop_sequence(self, by_stop):
    #     """A stop_sequence names a stop only within a shape, so check it there.

    #         left as todo for now - need a cassette with two shape ids
    #     """
    #     for _, at_shape in by_stop.groupby("shape_id", observed=True):
    #         assert set(at_shape["stop_sequence"]) == set(STOP_IDS)
    #         assert (
    #             at_shape["stop_id"] == at_shape["stop_sequence"].map(STOP_IDS)
    #         ).all()

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_one_row_per_stop_and_day_part_that_saw_an_arrival(self, by_stop):
        """A day part a stop saw no arrival in is absent, not present and empty."""
        assert not by_stop.duplicated(
            ["shape_id", "stop_sequence", "time_period"]
        ).any()
        assert (by_stop["n_timepoint_arrivals"] > 0).all()

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_one_row_per_day_part(self, by_period):
        assert not by_period.duplicated(["shape_id", "time_period"]).any()
        assert (by_period["n_timepoint_arrivals"] > 0).all()

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_the_two_tables_cover_the_same_day_parts(self, by_stop, by_period):
        assert set(by_period["time_period"]) == set(by_stop["time_period"])

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_the_pooled_count_is_the_sum_of_the_stop_counts(self, by_stop, by_period):
        """Every timepoint arrival is counted once in each table, and no more.

        sum(n_timepoint_arrivals) matches between the by_stop and by_period
        tables, within a shape -- the pooled table drops the stop from the grain
        and nothing else.
        """
        grain = ["shape_id", "time_period"]
        pooled = by_stop.groupby(grain, observed=True)["n_timepoint_arrivals"].sum()

        pd.testing.assert_series_equal(
            by_period.set_index(grain)["n_timepoint_arrivals"],
            pooled,
            check_names=False,
        )


class TestIdentities:
    """Relationships between the reported columns, on every cassette.

    These hold whatever the service did, so they run on variable_headway too --
    it is an hour out from its own schedule and a poor fixture for pinning a
    value, but the messier the input the better for an identity.
    """

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_proportions_are_proportions(self, by_stop, by_period):
        for frame in [by_stop, by_period]:
            for column in [*PROPORTION_COLUMNS, "schedule_adherence"]:
                assert frame[column].between(0.0, 1.0).all(), column
                assert frame[column].notna().all(), column

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_not_on_time_is_early_or_late(self, by_stop, by_period):
        """A vehicle cannot be both, so the two proportions add rather than overlap.

        Stated as an identity because it is the one column of the five that is
        derived rather than measured, and the derivation is the whole of it.
        """
        for frame in [by_stop, by_period]:
            pd.testing.assert_series_equal(
                frame["proportion_not_on_time"],
                frame["proportion_early"] + frame["proportion_late"],
                check_names=False,
            )

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_adherence_is_the_complement_of_not_on_time(self, by_stop, by_period):
        """The two columns are one number reported from both ends.

        Nothing is measured twice here -- schedule adherence exists so a table
        can lead with the share of service that made its published time -- so the
        test that earns its place is the one saying the pair always adds to one.
        """
        for frame in [by_stop, by_period]:
            pd.testing.assert_series_equal(
                frame["schedule_adherence"],
                1 - frame["proportion_not_on_time"],
                check_names=False,
            )

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_the_percentiles_are_ordered_and_non_negative(self, by_stop, by_period):
        """p50 cannot beat p80, and an absolute deviation cannot be negative.

        The ordering is the half with teeth: punctuality_thresholds spreads its
        deviations far enough apart that the two come back 30.0 and 204.4, so a
        run reporting them the wrong way round fails here.

        Non-negativity is a guard rail rather than a test of the absolute value.
        No cassette runs more than half of its arrivals early, so even a metric
        that dropped the ``abs`` would report a non-negative median on all seven.
        What catches a missing ``abs`` is inconsistent_feed's exact p80 of 84.0,
        which comes back 0.0 without it.
        """
        for frame in [by_stop, by_period]:
            assert (frame["p50_absolute_deviation_seconds"] >= 0).all()
            assert (
                frame["p50_absolute_deviation_seconds"]
                <= frame["p80_absolute_deviation_seconds"]
            ).all()


class TestServiceRunToSchedule:
    @pytest.mark.parametrize("cassette_name", PERFECT_CASSETTES)
    def test_hitting_every_published_arrival_reports_nothing(self, by_stop, by_period):
        """Every vehicle reached every timepoint exactly when it was published to, so there is no schedule deviation."""
        for frame in [by_stop, by_period]:
            assert set(frame["p50_absolute_deviation_seconds"]) == {0.0}
            assert set(frame["p80_absolute_deviation_seconds"]) == {0.0}
            for column in PROPORTION_COLUMNS:
                assert set(frame[column]) == {0.0}, column
            assert set(frame["schedule_adherence"]) == {1.0}

    @pytest.mark.parametrize("cassette_name", ["two_service_dates"])
    def test_a_trip_published_past_midnight_is_still_on_time(self, by_stop, by_period):
        """Check that times past midnight do not cause issues

        dim_stop_arrivals uses GTFS time, but fct_stop_time_updates uses clock time,
        this checks that this conversion is handled.
        """
        assert set(by_stop["p80_absolute_deviation_seconds"]) == {0.0}
        assert set(by_period["p80_absolute_deviation_seconds"]) == {0.0}

    @pytest.mark.parametrize("cassette_name", ["two_service_dates"])
    def test_the_day_part_follows_the_published_arrival(self, by_stop, by_period):
        """A trip published as 24:00 is reported in Owl the following morning.

        Ensures that dates are associated with calendar dates, rather than service dates.
        """
        for sequence in TIMEPOINT_SEQUENCES:
            at_stop = rows_for(by_stop, stop_sequence=sequence)
            assert (
                at_stop.set_index("time_period")["n_timepoint_arrivals"].to_dict()
                == BOUNDARY_PERIODS
            )

        assert by_period.set_index("time_period")["n_timepoint_arrivals"].to_dict() == {
            period: count * len(TIMEPOINT_SEQUENCES)
            for period, count in BOUNDARY_PERIODS.items()
        }


class TestSomeLateSomeEarly:
    """mixed_punctuality: the stop-level table earning its keep.

    Ten trips against their own timetable. Three run six minutes late at every
    timepoint. Three run to time all the way down the line and then reach the
    last timepoint two minutes early. So the last timepoint is the only place the
    early trips show up, and a metric reported only per day part would lose them
    in the pooling.
    """

    @pytest.mark.parametrize("cassette_name", ["mixed_punctuality"])
    @pytest.mark.parametrize("sequence", BEFORE_LAST_TIMEPOINT)
    def test_before_the_last_timepoint_only_the_late_trips_show(
        self, by_stop, sequence
    ):
        """Seven trips dead on time and three 360 s late, so three tenths are late.

        360 s beats the five minute threshold. Nothing is early here -- the early
        trips have not gone early yet -- and the absolute deviations are seven
        zeros and three 360s, which puts the median on zero and the 80th
        percentile inside the 360s.
        """
        assert_reports(one_row(by_stop, stop_sequence=sequence), MIXED_BEFORE_LAST)

    @pytest.mark.parametrize("cassette_name", ["mixed_punctuality"])
    def test_the_last_timepoint_sees_both(self, by_stop):
        """The same three late trips, plus three 120 s early arrivals.

        Two minutes early beats the one minute threshold, so the early proportion
        goes from nothing to three tenths at this stop and this stop only. The
        median moves with it: the sorted absolute deviations are four zeros,
        three 120s and three 360s, so the 50th percentile of ten values lands
        halfway between the fifth and sixth -- both 120 s -- where before the
        last timepoint it sat on zero.
        """
        assert_reports(one_row(by_stop, stop_sequence=LAST_TIMEPOINT), MIXED_AT_LAST)

    @pytest.mark.parametrize("cassette_name", ["mixed_punctuality"])
    def test_the_last_timepoint_differs_from_the_ones_before_it(self, by_stop):
        """Stated as a difference, so the two rows cannot silently converge.

        If the metric ever summarised a trip once rather than once per stop --
        calling the early trips early everywhere, or nowhere -- these three rows
        would come out identical and every value assertion above would still be
        checking the same number three times. Both reported columns that move
        are checked, because the proportion and the percentile can be got wrong
        independently.
        """
        at_stop = by_stop.set_index("stop_sequence")

        assert at_stop["proportion_early"][LAST_TIMEPOINT] > 0
        assert (at_stop["proportion_early"][BEFORE_LAST_TIMEPOINT] == 0).all()

        p50 = at_stop["p50_absolute_deviation_seconds"]
        assert p50[LAST_TIMEPOINT] > p50[BEFORE_LAST_TIMEPOINT].max()

    @pytest.mark.parametrize("cassette_name", ["mixed_punctuality"])
    def test_the_whole_day_part_pools_all_three_stops(self, by_period):
        """30 arrivals, of which three are early and nine are late.

        The three late trips are late at all three timepoints and the three early
        ones are early at just the one, so the pooled proportions are 9/30 and
        3/30. This is the only place in the suite that separates counting
        arrivals from counting trips: a metric classifying each *trip* once and
        pooling that would report 3/10 and 3/10 here, and every other cassette
        moves a whole trip together, so the two ways of counting agree
        everywhere else and nothing else would notice.
        """
        assert_reports(one_row(by_period, time_period="Early AM"), MIXED_POOLED)


class TestTheThresholds:
    """punctuality_thresholds: which side of each threshold the boundary falls on.

    Eight trips deviating by 0, -60, -61, +300, +301, 0, 0, 0 seconds, the same
    at every timepoint. Two land exactly on a threshold and two land a second
    past one, so this is the cassette that says whether the comparisons are
    inclusive.
    """

    @pytest.mark.parametrize("cassette_name", ["punctuality_thresholds"])
    @pytest.mark.parametrize("sequence", TIMEPOINT_SEQUENCES)
    def test_every_timepoint_reports_the_same_split(self, by_stop, sequence):
        """Two early, two late, and a p50 and p80 that both interpolate.

        A whole trip moves together here, so its classification is the same all
        the way down the line and all three stops report identically. The
        deviations are spread wide enough that neither percentile lands on a
        repeated zero:

            sorted |deviation| = 0, 0, 0, 0, 60, 61, 300, 301
            p50 = (value[3] + value[4]) / 2               = (0 + 60) / 2 = 30.0
            p80 = value[5] + 0.6 * (value[6] - value[5])
                = 61 + 0.6 * 239                                        = 204.4
        """
        assert_reports(one_row(by_stop, stop_sequence=sequence), THRESHOLD_BY_STOP)

    @pytest.mark.parametrize("cassette_name", ["punctuality_thresholds"])
    def test_exactly_on_a_threshold_is_not_on_time(self, by_stop):
        """Both comparisons are inclusive, so the boundary trips are counted.

        Four of the eight trips deviate: 60 s early, 61 s early, 300 s late and
        301 s late. All four are reported. "A minute early or more" takes in the
        trip exactly a minute early, and "five minutes late or more" takes in the
        one exactly five minutes late -- so half the service misses its published
        time and a quarter of it is early or late.

        This is the whole of what the cassette exists to pin. A ``<`` in place of
        a ``<=`` halves both proportions and nothing else in the suite notices.
        """
        row = one_row(by_stop, stop_sequence=LAST_TIMEPOINT)

        assert row["proportion_early"] == pytest.approx(2 / THRESHOLD_TRIPS)
        assert row["proportion_late"] == pytest.approx(2 / THRESHOLD_TRIPS)
        assert row["proportion_not_on_time"] == pytest.approx(4 / THRESHOLD_TRIPS)

    @pytest.mark.parametrize("cassette_name", ["punctuality_thresholds"])
    def test_the_pooled_percentile_is_not_the_average_of_the_stops(self, by_period):
        """Pooling is over arrivals, not over the per-stop numbers.

        All three stops report a p80 of 204.4 s, so any averaging of them --
        weighted or not -- gives 204.4 back. Pooling the 24 arrivals instead puts
        the 80th percentile at the 18.4th sorted value, which is inside the three
        300s: 12 zeros, three 60s, three 61s, three 300s, three 301s.

        The two tables answer different questions and this is the case that
        proves it, because everything else about the three stops is identical.
        """
        pooled = one_row(by_period, time_period="Early AM")

        assert_reports(pooled, THRESHOLD_POOLED)
        assert pooled["p80_absolute_deviation_seconds"] != pytest.approx(
            THRESHOLD_BY_STOP["p80_absolute_deviation_seconds"]
        )


class TestAFeedThatContradictsItself:
    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_an_impossible_arrival_is_reported_as_early_running(self, by_stop):
        """A stop time the feed got wrong is indistinguishable from a fast trip.

        inconsistent_feed's second trip on each date reports reaching stop 2
        twenty seconds before it left stop 1 -- 140 s before the schedule said it
        would. Travel time variability drops that observation, because a negative
        travel time is not a valid travel time. On time performance keeps it: an
        arrival 140 s ahead of the published one is more than a minute early, we
        cannot tell at which trip the error occurs at.

        Deliberate, but temporary until we can get more reliable stop arrivals. Two
        of the eight arrivals at stop 2 are affected, and the sorted absolute
        deviations there are six zeros and two 140s, so
            p80 = value[5] + 0.6 * (value[6] - value[5]) = 0 + 0.6 * 140 = 84.0

        """
        row = one_row(by_stop, stop_sequence=DISORDERED_SEQUENCE)

        assert row["proportion_early"] == pytest.approx(
            DISORDERED_TRIPS / INCONSISTENT_TRIPS
        )
        assert row["proportion_late"] == 0.0
        assert row["p80_absolute_deviation_seconds"] == pytest.approx(
            0.6 * abs(DISORDERED_DEVIATION_SECONDS)
        )

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_the_deviation_stays_at_the_stop_it_happened_at(self, by_stop):
        """Only the arrival at stop 2 moved, so only stop 2 reports anything.

        The trip's *departure* from stop 2 is where it always was, so it reaches
        stop 5 exactly on time.

        This is the only cassette where a stop in the middle of the line
        deviates. mixed_punctuality's early trips go early at the last timepoint,
        which has nothing after it to smear into, and every other deviation in
        the suite moves a whole trip together. So a metric that carried a
        deviation forward -- measuring against the previous timepoint's arrival
        rather than the published one, or spreading a trip's worst stop across
        the rest of its run -- is caught here and nowhere else.
        """
        unaffected = by_stop.loc[by_stop["stop_sequence"] != DISORDERED_SEQUENCE]

        assert unaffected["stop_sequence"].to_list() == [1, LAST_TIMEPOINT]
        assert set(unaffected["p80_absolute_deviation_seconds"]) == {0.0}
        for column in PROPORTION_COLUMNS:
            assert set(unaffected[column]) == {0.0}, column


class TestThresholdArguments:
    """The thresholds are arguments, and moving one has to move the answer."""

    def test_a_wider_late_threshold_forgives_the_late_trips(self):
        """mixed_punctuality's late trips are 360 s down, so 400 s clears them.

        Nothing else changes: the early trips are still early at the last
        timepoint, so that stop keeps three tenths not on time while the two
        before it drop to nothing.
        """
        result = run_on_time_performance(
            "mixed_punctuality", late_seconds=LATE_THRESHOLD_SECONDS + 100
        )

        assert set(result.by_stop["proportion_late"]) == {0.0}
        late_stop = one_row(result.by_stop, stop_sequence=LAST_TIMEPOINT)
        early = MIXED_EARLY_TRIPS / MIXED_TRIPS
        assert late_stop["proportion_early"] == pytest.approx(early)
        assert late_stop["proportion_not_on_time"] == pytest.approx(early)

    def test_a_wider_early_threshold_forgives_the_early_arrivals(self):
        """The 120 s early arrivals are the only early ones, so 180 s clears them."""
        result = run_on_time_performance(
            "mixed_punctuality", early_seconds=EARLY_THRESHOLD_SECONDS + 120
        )

        assert set(result.by_stop["proportion_early"]) == {0.0}
        assert set(result.by_stop["proportion_late"]) == {
            MIXED_LATE_TRIPS / MIXED_TRIPS
        }

    def test_loosening_a_threshold_forgives_the_boundary_trip(self):
        """61 s early and 301 s late forgive the trips sitting on the defaults.

        The reverse of the inclusiveness test: the two trips that are not on time
        at 60 and 300 are on time a second either side of them, which pins the
        comparison from the other direction. Only the trips a second past each
        moved threshold are left, one on each side.
        """
        result = run_on_time_performance(
            "punctuality_thresholds",
            early_seconds=EARLY_THRESHOLD_SECONDS + 1,
            late_seconds=LATE_THRESHOLD_SECONDS + 1,
        )
        row = one_row(result.by_stop, stop_sequence=LAST_TIMEPOINT)

        assert row["proportion_early"] == pytest.approx(1 / THRESHOLD_TRIPS)
        assert row["proportion_late"] == pytest.approx(1 / THRESHOLD_TRIPS)
        assert row["proportion_not_on_time"] == pytest.approx(2 / THRESHOLD_TRIPS)

    def test_the_percentiles_name_the_columns(self):
        """The reported column names follow the percentiles asked for.

        Same contract as travel time variability's, so a run reporting something
        other than the default cannot be mistaken for one that did not.
        """
        result = run_on_time_performance("punctuality_thresholds", upper_percentile=0.9)

        for frame in [result.by_stop, result.by_period]:
            assert "p90_absolute_deviation_seconds" in frame.columns
            assert "p80_absolute_deviation_seconds" not in frame.columns
