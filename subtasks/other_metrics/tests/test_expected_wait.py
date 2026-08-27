"""Expected and excess wait, played end to end over the cassettes.

Each test names a cassette and says what the reported table should be. Nothing
in between is asserted: the loaders are private, so the run goes from parquet
straight to ``calculate_expected_wait``'s output.

None of the cassettes have any jitter, so every value below is exact and derived
from the service the fixture describes rather than read back off a previous run.
Where a rounded literal appears it carries its arithmetic in the docstring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.metric_calculation import TIME_PERIODS
from tests.conftest import CASSETTES

# headways are reported at every stop, timepoint or not
ALL_SEQUENCES = [1, 2, 3, 4, 5]
STOP_IDS = {1: "stop1", 2: "stop2", 3: "stop3", 4: "stop4", 5: "stop5"}

# the day parts, as specified, in the order the clock runs through them. stated
# here independently of the implementation, so moving a boundary has to be done
# in both places
EXPECTED_TIME_PERIODS = [
    ("Owl", 0, 4),
    ("Early AM", 4, 7),
    ("AM Peak", 7, 10),
    ("Midday", 10, 15),
    ("PM Peak", 15, 20),
    ("Evening", 20, 24),
]
PERIOD_NAMES = [name for name, _, _ in EXPECTED_TIME_PERIODS]

# every cassette publishes the even 15 minute grid. a headway is the gap between
# vehicles -- one departure to the next -- so a 15 minute service publishes the
# full 900 s whatever its dwell, and half of that is 7:30
SCHEDULED_HEADWAY_SECONDS = 900.0
SCHEDULED_EXPECTED_WAIT_SECONDS = SCHEDULED_HEADWAY_SECONDS / 2  # 450.0

# the cassettes whose service ran exactly to that schedule, everywhere.
# inconsistent_feed is one of them only because a headway is measured departure
# to departure: the arrival it contradicts itself about is not looked at, so the
# bad stop time cannot reach this metric at all
EVEN_SERVICE_CASSETTES = [
    "constant_headway",
    "two_service_dates",
    "inconsistent_feed",
]

# variable_headway's two blocks
SHORT_HEADWAY_SECONDS = 600.0  # 10 minutes
LONG_HEADWAY_SECONDS = 1200.0  # 20 minutes
EVEN_HEADWAY_SECONDS = 900.0  # 15 minutes -- and exactly what the schedule says

# alternating 10 and 20 averages the same 15 minutes Midday runs evenly, so the
# two blocks report an identical mean headway and differ only in how evenly it
# was spread. A trip runs the line in 10 minutes, so both sit clear of their day
# part's boundary and these hold identically at all five stops
ALTERNATING_PERIOD = "AM Peak"
EVEN_PERIOD = "Midday"
ALTERNATING_GAPS = {SHORT_HEADWAY_SECONDS: 6, LONG_HEADWAY_SECONDS: 6}  # n = 12
EVEN_GAPS = {EVEN_HEADWAY_SECONDS: 11}  # n = 11

# how the gaps at one stop fall across the day parts, cassette by cassette. every
# stop sees the same split, because no cassette has a block boundary that a trip
# crosses while running the line
GAPS_PER_STOP = {
    "constant_headway": {"Early AM": 4, "AM Peak": 3},
    "variable_headway": {"AM Peak": 12, "Midday": 11},
    "variable_travel_time": {"Early AM": 4, "AM Peak": 4},
    "two_service_dates": {"Owl": 1, "Early AM": 3, "Evening": 2},
    "inconsistent_feed": {"Early AM": 6},  # 3 on each of its two dates
    # both punctuality cassettes run early enough in the morning that every gap
    # opens inside Early AM, however late their late trips are
    "mixed_punctuality": {"Early AM": 9},
    "punctuality_thresholds": {"Early AM": 7},
}

# variable_travel_time is a travel time fixture: its fast trips catch the ones in
# front of them, so past the first stop its headways are deliberately uneven and
# there is no clean expected wait to pin. It still has to satisfy the schema and
# the identities, so it stays in those tests and is left out of the value ones
WAIT_CASSETTES = [name for name in CASSETTES if name != "variable_travel_time"]

EXPECTED_COLUMNS = [
    "shape_id",
    "stop_sequence",
    "stop_id",
    "time_period",
    "n_headways",
    "mean_rt_headway_seconds",
    "median_rt_headway_seconds",
    "max_rt_headway_seconds",
    "rt_headway_variance",
    "mean_scheduled_headway_seconds",
    "median_scheduled_headway_seconds",
    "scheduled_expected_wait_seconds",
    "rt_expected_wait_seconds",
    "excess_wait_seconds",
    "percent_time_excess_wait",
]


class TestTimePeriods:
    """The day parts themselves, before anything is measured over them."""

    def test_periods_are_as_specified(self):
        assert [tuple(period) for period in TIME_PERIODS] == [
            tuple(period) for period in EXPECTED_TIME_PERIODS
        ]

    def test_every_hour_lands_in_exactly_one_period(self):
        """Contiguous and covering the whole clock, so nothing falls between them."""
        for hour in range(24):
            matching = [
                name for name, start, end in TIME_PERIODS if start <= hour < end
            ]
            assert len(matching) == 1, f"hour {hour} matched {matching}"


class TestOutputFormat:
    def test_schema(self, expected_wait):
        assert list(expected_wait.columns) == EXPECTED_COLUMNS

    def test_dtypes(self, expected_wait):
        assert expected_wait["shape_id"].dtype == object
        assert expected_wait["stop_id"].dtype == object
        assert expected_wait["stop_sequence"].dtype.kind == "i"
        assert expected_wait["n_headways"].dtype.kind == "i"
        for column in EXPECTED_COLUMNS[5:]:
            assert expected_wait[column].dtype.kind == "f", column

    def test_index_is_reset(self, expected_wait):
        # the grain comes back as columns rather than a MultiIndex, so the
        # notebook can merge the geometry straight onto it
        assert expected_wait.index.to_list() == list(range(len(expected_wait)))


class TestTheGrain:
    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_one_row_per_stop_and_day_part_that_saw_service(
        self, expected_wait, cassette_name
    ):
        """Every stop and period that saw a headway gets a row, and nothing else.

        A period with no service at a stop must not appear at all -- grouping on
        a categorical will happily invent those rows, and they would carry a zero
        count and a NaN wait.
        """
        gaps = GAPS_PER_STOP[cassette_name]
        expected = [
            (sequence, period)
            for sequence in ALL_SEQUENCES
            for period in PERIOD_NAMES
            if period in gaps
        ]

        assert (
            list(
                expected_wait[["stop_sequence", "time_period"]].itertuples(index=False)
            )
            == expected
        )

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_every_headway_is_counted_exactly_once(self, expected_wait, cassette_name):
        """Test that sum(n_headways) = (n trips - 1) * n stops. Important because this counts over the entire day"""
        per_stop = sum(GAPS_PER_STOP[cassette_name].values())

        assert expected_wait["n_headways"].sum() == per_stop * len(ALL_SEQUENCES)
        assert (expected_wait["n_headways"] > 0).all()

    @pytest.mark.parametrize("cassette_name", ["inconsistent_feed"])
    def test_a_trip_id_reused_across_dates_does_not_pair_across_them(
        self, expected_wait
    ):
        """Real feeds reuse trip ids daily, and pairing is within a service date.

        This asserts the working rule that we do not track headways between service dates.
        """
        assert set(expected_wait["n_headways"]) == {6}

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_stop_id_rides_along_with_the_stop_sequence(self, expected_wait):
        """Check that all stop_ids match the expected stop sequence"""
        assert expected_wait.set_index("stop_sequence")["stop_id"].to_dict() == STOP_IDS


class TestIdentities:
    """Relationships between the reported columns, on every cassette.

    These hold whatever the service did, so they run on variable_travel_time too
    -- its headways make a poor fixture for pinning a value, but the messier the
    input the better for an identity.
    """

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_scheduled_expected_wait_is_half_the_median_scheduled_headway(
        self, expected_wait
    ):
        """scheduled expected wait is alwys 50% of scheduled headway"""
        assert (
            expected_wait["scheduled_expected_wait_seconds"]
            == expected_wait["median_scheduled_headway_seconds"] / 2
        ).all()

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_every_cassette_is_scheduled_on_the_same_even_grid(self, expected_wait):
        """900 s between vehicles, so 450 s of expected wait.

        The dwell does not enter into it. A headway is one departure to the
        next, so a 15 minute service publishes the full 15 minutes however long
        each bus stands at the stop.
        """
        assert set(expected_wait["median_scheduled_headway_seconds"]) == {
            SCHEDULED_HEADWAY_SECONDS
        }
        assert set(expected_wait["mean_scheduled_headway_seconds"]) == {
            SCHEDULED_HEADWAY_SECONDS
        }
        assert set(expected_wait["scheduled_expected_wait_seconds"]) == {
            SCHEDULED_EXPECTED_WAIT_SECONDS
        }

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_excess_wait_is_the_difference_between_the_two_expected_waits(
        self, expected_wait
    ):
        """Excess wait is what the rider got, less what the schedule promised."""
        difference = (
            expected_wait["rt_expected_wait_seconds"]
            - expected_wait["scheduled_expected_wait_seconds"]
        )

        pd.testing.assert_series_equal(
            expected_wait["excess_wait_seconds"], difference, check_names=False
        )

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_percent_time_excess_wait_is_a_proportion(self, expected_wait):
        assert expected_wait["percent_time_excess_wait"].between(0.0, 1.0).all()
        assert expected_wait["percent_time_excess_wait"].notna().all()

    @pytest.mark.parametrize("cassette_name", CASSETTES)
    def test_time_is_spent_waiting_over_exactly_where_a_gap_ran_long(
        self, expected_wait
    ):
        """``max(H - S, 0)`` is positive for some gap if and only if some H beat S.

        Both directions matter. Service landing exactly on the published headway
        contributes nothing -- ``max(870 - 870, 0)`` is zero -- so a day part
        whose longest gap is the scheduled one must report no time spent waiting
        over, and one with a longer gap must report some. Every cassette
        publishes a single scheduled headway, so its longest gap is what decides
        this.
        """
        ran_long = (
            expected_wait["max_rt_headway_seconds"]
            > expected_wait["median_scheduled_headway_seconds"]
        )

        pd.testing.assert_series_equal(
            expected_wait["percent_time_excess_wait"] > 0, ran_long, check_names=False
        )


class TestServiceRunToSchedule:
    @pytest.mark.parametrize("cassette_name", EVEN_SERVICE_CASSETTES)
    def test_even_service_waits_exactly_the_scheduled_wait(self, expected_wait):
        """An even 15 minutes, run exactly to schedule, at every stop and period.

        Every gap is 870 s, so the variance term is zero and the realtime
        expected wait collapses to half the headway -- the same 435 s the
        schedule promised, with no excess and no time spent waiting over.

        Skips the day parts holding a single gap, which have no sample variance;
        those are pinned separately.
        """
        measured = expected_wait.loc[expected_wait["n_headways"] > 1]

        assert set(measured["mean_rt_headway_seconds"]) == {SCHEDULED_HEADWAY_SECONDS}
        assert set(measured["max_rt_headway_seconds"]) == {SCHEDULED_HEADWAY_SECONDS}
        assert set(measured["rt_headway_variance"]) == {0.0}
        assert set(measured["rt_expected_wait_seconds"]) == {
            SCHEDULED_EXPECTED_WAIT_SECONDS
        }
        assert set(measured["excess_wait_seconds"]) == {0.0}
        assert set(measured["percent_time_excess_wait"]) == {0.0}

    @pytest.mark.parametrize("cassette_name", ["two_service_dates"])
    def test_a_service_date_boundary_does_not_leak_into_the_metric(self, expected_wait):
        """The same even service, split over two dates, still reports 870 s gaps.

        two_service_dates runs 23:30 to 24:15 on one date and 04:00 to 04:45 on
        the next, so its trips cross midnight and are published in GTFS hours
        past 24. None of that reaches the metric: the overnight break is not
        reported as a headway at all, and every gap that is reported -- the
        23:45 to 24:00 pair included -- is the usual 870 s.
        """
        assert set(expected_wait["time_period"]) == {"Evening", "Owl", "Early AM"}
        assert set(expected_wait["mean_rt_headway_seconds"]) == {
            SCHEDULED_HEADWAY_SECONDS
        }

    @pytest.mark.parametrize("cassette_name", ["two_service_dates"])
    def test_a_single_gap_has_no_realtime_expected_wait(self, expected_wait):
        """One observation says nothing about how evenly service ran.

        Owl holds a single gap, so it has no sample variance and the realtime
        expected wait comes back NaN rather than half the one gap. Deliberate
        behaviour, and this is the day part that exercises it -- the scheduled
        side is unaffected, since it needs no variance term.
        """
        owl = expected_wait.loc[expected_wait["time_period"] == "Owl"]

        assert owl["n_headways"].to_list() == [1] * len(ALL_SEQUENCES)
        assert owl["rt_headway_variance"].isna().all()
        assert owl["rt_expected_wait_seconds"].isna().all()
        assert owl["excess_wait_seconds"].isna().all()
        # the gap itself is still reported, and still counts toward time waited
        assert set(owl["mean_rt_headway_seconds"]) == {SCHEDULED_HEADWAY_SECONDS}
        assert set(owl["scheduled_expected_wait_seconds"]) == {
            SCHEDULED_EXPECTED_WAIT_SECONDS
        }
        assert set(owl["percent_time_excess_wait"]) == {0.0}


class TestServiceThatDidNot:
    """variable_headway: the same mean headway spread two different ways."""

    @pytest.mark.parametrize("cassette_name", ["variable_headway"])
    def test_midday_runs_exactly_to_schedule(self, expected_wait):
        """A constant 15 minutes, which is what the schedule published.

        Every gap is 870 s, so there is no variance term and the expected wait
        collapses to half the headway -- the same 435 s the schedule promised,
        and no excess. Identical at all five stops, because Midday sits clear of
        both ends of the constant block.
        """
        rows = expected_wait.loc[expected_wait["time_period"] == EVEN_PERIOD]

        assert len(rows) == len(ALL_SEQUENCES)
        assert set(rows["n_headways"]) == {sum(EVEN_GAPS.values())}
        assert set(rows["mean_rt_headway_seconds"]) == {EVEN_HEADWAY_SECONDS}
        assert set(rows["median_rt_headway_seconds"]) == {EVEN_HEADWAY_SECONDS}
        assert set(rows["max_rt_headway_seconds"]) == {EVEN_HEADWAY_SECONDS}
        assert set(rows["rt_headway_variance"]) == {0.0}
        assert set(rows["rt_expected_wait_seconds"]) == {
            SCHEDULED_EXPECTED_WAIT_SECONDS
        }
        assert set(rows["excess_wait_seconds"]) == {0.0}
        assert set(rows["percent_time_excess_wait"]) == {0.0}

    @pytest.mark.parametrize("cassette_name", ["variable_headway"])
    def test_am_peak_alternates(self, expected_wait):
        """The same mean headway as Midday, and a 56 s longer wait.

        AM Peak holds 6 gaps of 570 s and 6 of 1170 s, so it averages the very
        870 s that Midday runs evenly and that the schedule publishes. Every
        headline number about the service is identical. The rider's wait is not,
        because they are twice as likely to turn up during a long gap:

            n     = 12  (6 gaps of 600 s, 6 of 1200 s)
            mean  = (6*600 + 6*1200) / 12 = 10800 / 12   = 900.0
            var   = (6*(600-900)^2 + 6*(1200-900)^2) / 11
                  = (6*90000 + 6*90000) / 11 = 1080000/11 = 98181.818182
            E[W]  = mean/2 + var/(2*mean)
                  = 450 + 1080000 / (11 * 2 * 900)
                  = 450 + 54.545455                       = 504.545455

        So the excess wait is 54.545455 s against Midday's zero, on service whose
        mean headway, median headway and published schedule all match it exactly.
        That is the whole point of the metric: the mean cannot see the difference
        and the rider can.
        """
        rows = expected_wait.loc[expected_wait["time_period"] == ALTERNATING_PERIOD]

        assert len(rows) == len(ALL_SEQUENCES)
        assert set(rows["n_headways"]) == {sum(ALTERNATING_GAPS.values())}
        # the mean and median are both 870, exactly as in Midday
        assert set(rows["mean_rt_headway_seconds"]) == {EVEN_HEADWAY_SECONDS}
        assert set(rows["median_rt_headway_seconds"]) == {EVEN_HEADWAY_SECONDS}
        assert set(rows["max_rt_headway_seconds"]) == {LONG_HEADWAY_SECONDS}

        # approx does not broadcast a scalar over a list, so state the value once
        # per stop -- all five carry it
        def at_every_stop(value):
            return pytest.approx([value] * len(ALL_SEQUENCES))

        assert rows["rt_headway_variance"].to_list() == at_every_stop(1080000 / 11)
        assert rows["rt_expected_wait_seconds"].to_list() == at_every_stop(
            450 + 1080000 / (11 * 2 * 900)
        )
        assert rows["excess_wait_seconds"].to_list() == pytest.approx(
            [54.545455] * len(ALL_SEQUENCES), abs=1e-6
        )

    @pytest.mark.parametrize("cassette_name", ["variable_headway"])
    def test_am_peak_excess_share(self, expected_wait):
        """Half the gaps run long, but only a sixth of the time is spent waiting over.

        A rider turns up at a random moment inside a gap and waits out whatever
        is left of it, so what they experience is not the gap's length but their
        own remaining wait. A short gap never leaves anyone waiting more than
        900 s. A long one does -- but only for the riders arriving in its first
        1200 - 900 = 300 seconds; someone turning up in the last 900 s of that
        same gap still gets a bus inside the scheduled headway.

            over-long = 6 * max(1200 - 900, 0) + 6 * max(600 - 900, 0)
                      = 6 * 300 + 6 * 0                            =  1800
            total     = 6 * 600 + 6 * 1200                         = 10800
            share     = 1800 / 10800 = 1/6                         = 0.166667

        Note what the clamp is doing. Without it the short gaps would score
        6 * (600 - 900) = -1800 and cancel the long ones exactly, reporting 0 for
        a day part that is anything but even. AM Peak is the case that catches
        that, and neither even cassette would.
        """
        rows = expected_wait.loc[expected_wait["time_period"] == ALTERNATING_PERIOD]
        shares = rows["percent_time_excess_wait"].to_list()

        assert len(rows) == len(ALL_SEQUENCES)
        assert shares == pytest.approx([1800 / 10800] * len(ALL_SEQUENCES))
        assert shares == pytest.approx([1 / 6] * len(ALL_SEQUENCES))
        assert shares == pytest.approx([0.166667] * len(ALL_SEQUENCES), abs=1e-6)
