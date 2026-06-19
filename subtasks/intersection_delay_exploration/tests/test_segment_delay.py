"""Unit tests for the segment delay breakdown (segment_delay.py)."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString

import segment_delay as sd


# --- classify_trip_period ---

@pytest.mark.parametrize(
    "hour, expected",
    [(22, "night"), (21, "night"), (2, "night"), (5, "night"), (6, "day"), (12, "day"), (20, "day")],
)
def test_classify_trip_period(hour, expected):
    assert sd.classify_trip_period(pd.Timestamp(f"2026-02-03 {hour:02d}:30:00")) == expected


# --- build_segments ---

def test_build_segments_drops_segments_ending_at_nearside_signal():
    signals = pd.Series([0.0, 100.0, 250.0, 400.0], index=[0, 1, 2, 3])
    nearside = pd.Series([False, True, False, False], index=[0, 1, 2, 3])

    segments = sd.build_segments(signals, nearside)

    # (0->1) ends at near-side signal 1 -> dropped; (1->2) and (2->3) kept
    assert list(segments.index) == ["sig1_to_sig2", "sig2_to_sig3"]
    assert segments.loc["sig1_to_sig2", "start_distance_m"] == 100.0
    assert segments.loc["sig2_to_sig3", "length_m"] == 150.0


def test_build_segments_orders_by_distance():
    signals = pd.Series([400.0, 0.0, 200.0], index=[2, 0, 1])  # unsorted
    nearside = pd.Series(False, index=[0, 1, 2])
    segments = sd.build_segments(signals, nearside)
    assert list(segments["start_distance_m"]) == [0.0, 200.0]


# --- _signal_delay_components ---

def _make_speeds(creep_until=151):
    """Constant creeping speed series indexed by seconds-into-trip."""
    return pd.Series(sd.FASTER_PERIOD_SPEED_MPS / 3, index=np.arange(0, creep_until, 1.0))


def test_signal_delay_uniform_and_connected_overflow():
    downstream = 1000.0
    stops = pd.DataFrame([
        dict(end_distance_m=downstream - 20, duration_s=30, start_time_s=100, end_time_s=130),
        dict(end_distance_m=downstream - (sd.SIGNAL_STOP_AREA_M + 50), duration_s=15, start_time_s=70, end_time_s=85),
    ])
    speeds = _make_speeds()  # always creeping -> upstream stop connected as overflow

    uniform, overflow, has_signal = sd._signal_delay_components(stops, speeds, downstream)
    assert (uniform, overflow, has_signal) == (30.0, 15.0, True)


def test_signal_delay_faster_period_breaks_overflow_chain():
    downstream = 1000.0
    stops = pd.DataFrame([
        dict(end_distance_m=downstream - 20, duration_s=30, start_time_s=100, end_time_s=130),
        dict(end_distance_m=downstream - (sd.SIGNAL_STOP_AREA_M + 50), duration_s=15, start_time_s=70, end_time_s=85),
        dict(end_distance_m=downstream - (sd.SIGNAL_STOP_AREA_M + 80), duration_s=10, start_time_s=40, end_time_s=55),
    ])
    speeds = _make_speeds()
    speeds.loc[58:67] = sd.FASTER_PERIOD_SPEED_MPS * 2  # faster period between stop2 and stop1

    uniform, overflow, has_signal = sd._signal_delay_components(stops, speeds, downstream)
    # only the first upstream stop is overflow; the one past the faster period is congestion
    assert (uniform, overflow, has_signal) == (30.0, 15.0, True)


def test_signal_delay_none_when_nearest_stop_outside_signal_area():
    downstream = 1000.0
    stops = pd.DataFrame([
        dict(end_distance_m=downstream - (sd.SIGNAL_STOP_AREA_M + 30), duration_s=20, start_time_s=10, end_time_s=30),
    ])
    assert sd._signal_delay_components(stops, _make_speeds(), downstream) == (0.0, 0.0, False)


def test_signal_delay_no_stops_returns_zero():
    assert sd._signal_delay_components(
        pd.DataFrame(columns=["end_distance_m", "duration_s", "start_time_s", "end_time_s"]),
        _make_speeds(),
        1000.0,
    ) == (0.0, 0.0, False)


def test_classify_segment_stops_labels_all_types():
    downstream = 1000.0
    stops = pd.DataFrame([
        dict(end_distance_m=downstream - 20, duration_s=30, start_time_s=100, end_time_s=130),
        dict(end_distance_m=downstream - (sd.SIGNAL_STOP_AREA_M + 50), duration_s=15, start_time_s=70, end_time_s=85),
        dict(end_distance_m=downstream - 500, duration_s=12, start_time_s=20, end_time_s=32),
        dict(end_distance_m=downstream - 700, duration_s=20, start_time_s=5, end_time_s=25),
    ], index=[0, 1, 2, 3])
    is_dwell = np.array([False, False, False, True])

    labels = sd.classify_segment_stops(stops, is_dwell, _make_speeds(), downstream)

    assert list(labels) == ["uniform", "overflow", "congestion", "dwell"]


def test_example_trip_segments_picks_one_per_flavor():
    daytime = pd.DataFrame([
        dict(service_date="d", TRIP_KEY="t1", segment_id="s1", uniform_delay_s=10, overflow_delay_s=0, signal_delay_s=10, congestion_delay_s=0),
        dict(service_date="d", TRIP_KEY="t2", segment_id="s2", uniform_delay_s=5, overflow_delay_s=8, signal_delay_s=13, congestion_delay_s=2),
        dict(service_date="d", TRIP_KEY="t3", segment_id="s3", uniform_delay_s=0, overflow_delay_s=0, signal_delay_s=0, congestion_delay_s=20),
    ])
    examples = sd.example_trip_segments(daytime).set_index("example")

    assert examples.loc["signal (uniform)", "TRIP_KEY"] == "t1"
    assert examples.loc["overflow", "TRIP_KEY"] == "t2"
    assert examples.loc["congestion", "TRIP_KEY"] == "t3"


# --- baselines ---

def _table(rows):
    columns = [
        "service_date", "TRIP_KEY", "period", "segment_id", "observed_travel_time_s",
        "dwell_s", "uniform_stop_duration_s", "overflow_stop_duration_s", "has_signal_stop",
    ]
    return pd.DataFrame(rows, columns=columns)


def _night_travel_row(tt, segment="S"):
    return dict(service_date="d", TRIP_KEY=f"n{tt}", period="night", segment_id=segment,
                observed_travel_time_s=tt, dwell_s=0, uniform_stop_duration_s=0,
                overflow_stop_duration_s=0, has_signal_stop=False)


def test_free_flow_uses_configured_percentile():
    travel_times = [40, 42, 44, 46, 48]
    table = _table([_night_travel_row(tt) for tt in travel_times])
    tff = sd.compute_free_flow_travel_times(table)
    assert tff["S"] == pytest.approx(np.percentile(travel_times, sd.FREE_FLOW_PERCENTILE))


def test_baseline_is_nan_below_min_sample_size():
    table = _table([_night_travel_row(tt) for tt in [40, 42]])  # only 2 < MIN_BASELINE_TRIPS
    assert pd.isna(sd.compute_free_flow_travel_times(table)["S"])


def test_red_phase_cap_uses_signal_stop_durations():
    durations = [25, 28, 30, 33, 40]
    rows = [
        dict(service_date="d", TRIP_KEY=f"s{d}", period="night", segment_id="S",
             observed_travel_time_s=100, dwell_s=0, uniform_stop_duration_s=d,
             overflow_stop_duration_s=0, has_signal_stop=True)
        for d in durations
    ]
    cap = sd.compute_red_phase_caps(_table(rows))
    assert cap["S"] == pytest.approx(np.percentile(durations, sd.RED_PHASE_PERCENTILE))


# --- decompose_delays ---

def test_decompose_caps_uniform_and_computes_residual_congestion():
    night = [_night_travel_row(tt) for tt in [40, 42, 44, 46, 48]]
    night += [
        dict(service_date="d", TRIP_KEY=f"s{d}", period="night", segment_id="S",
             observed_travel_time_s=100, dwell_s=0, uniform_stop_duration_s=d,
             overflow_stop_duration_s=0, has_signal_stop=True)
        for d in [25, 28, 30, 33, 40]
    ]
    day = [dict(service_date="d", TRIP_KEY="day1", period="day", segment_id="S",
                observed_travel_time_s=140, dwell_s=5, uniform_stop_duration_s=60,
                overflow_stop_duration_s=10, has_signal_stop=True)]
    table = _table(night + day)

    tff = sd.compute_free_flow_travel_times(table)
    cap = sd.compute_red_phase_caps(table)
    decomposed = sd.decompose_delays(table, tff, cap, period="day")
    row = decomposed.iloc[0]

    cap_value = cap["S"]
    expected_uniform = min(60, cap_value)
    expected_overflow = 10 + max(0, 60 - cap_value)
    expected_signal = expected_uniform + expected_overflow
    expected_congestion = max(0, 140 - tff["S"] - 5 - expected_signal)

    assert row["uniform_delay_s"] == pytest.approx(expected_uniform)
    assert row["overflow_delay_s"] == pytest.approx(expected_overflow)
    assert row["signal_delay_s"] == pytest.approx(expected_signal)
    assert row["congestion_delay_s"] == pytest.approx(expected_congestion)
    assert row["total_delay_s"] == pytest.approx(expected_signal + expected_congestion)


def test_decompose_congestion_clipped_at_zero():
    # fast day trip below free-flow -> no negative congestion
    night = [_night_travel_row(tt) for tt in [40, 42, 44, 46, 48]]
    day = [dict(service_date="d", TRIP_KEY="fast", period="day", segment_id="S",
                observed_travel_time_s=30, dwell_s=0, uniform_stop_duration_s=0,
                overflow_stop_duration_s=0, has_signal_stop=False)]
    table = _table(night + day)
    tff = sd.compute_free_flow_travel_times(table)
    cap = sd.compute_red_phase_caps(table)
    row = sd.decompose_delays(table, tff, cap, period="day").iloc[0]
    assert row["congestion_delay_s"] == 0.0
    assert row["signal_delay_s"] == 0.0


def test_decompose_uniform_uncapped_when_no_red_phase_estimate():
    # no night signal stops -> red-phase cap NaN -> uniform uncapped
    night = [_night_travel_row(tt) for tt in [40, 42, 44, 46, 48]]
    day = [dict(service_date="d", TRIP_KEY="day1", period="day", segment_id="S",
                observed_travel_time_s=140, dwell_s=0, uniform_stop_duration_s=60,
                overflow_stop_duration_s=0, has_signal_stop=True)]
    table = _table(night + day)
    tff = sd.compute_free_flow_travel_times(table)
    cap = sd.compute_red_phase_caps(table)
    assert pd.isna(cap.get("S"))
    row = sd.decompose_delays(table, tff, cap, period="day").iloc[0]
    assert row["uniform_delay_s"] == 60.0
    assert row["overflow_delay_s"] == 0.0


# --- segment_geometries ---

def test_segment_geometries_extracts_substring():
    shapes = gpd.GeoDataFrame(
        {"shape_id": ["X"]},
        geometry=[LineString([(0, 0), (1000, 0)])],
        crs="EPSG:3310",
    )
    segments = pd.DataFrame(
        [dict(start_distance_m=100, end_distance_m=400)], index=["seg1"]
    )
    geometries = sd.segment_geometries(shapes, "X", segments)
    assert geometries.crs == shapes.crs
    assert geometries.geometry.iloc[0].length == pytest.approx(300.0)
