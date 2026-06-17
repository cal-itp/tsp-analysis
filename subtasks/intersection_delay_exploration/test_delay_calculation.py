"""Unit tests for the stopped-event and near-side-signal helpers."""

import numpy as np
import pandas as pd

from delay_calculation import (
    MIN_STOPPED_RUN_SECONDS,
    STOPPED_SPEED_THRESHOLD_MPS,
    find_stopped_events,
    identify_nearside_signals,
)


def _trajectory(distances, freq_s=1.0):
    index = pd.date_range("2026-02-03 12:00:00", periods=len(distances), freq=f"{freq_s}s")
    return pd.Series(np.asarray(distances, dtype=float), index=index)


def test_find_stopped_events_reports_times_and_distances():
    distances = np.linspace(0, 200, 20)
    speeds = np.full(20, 5.0)
    speeds[5:13] = 0.0  # 8 consecutive slow samples -> 7s stop (> MIN_STOPPED_RUN_SECONDS)
    events = find_stopped_events(_trajectory(distances), pd.Series(speeds, index=_trajectory(distances).index))

    assert {"start_time_s", "end_time_s", "duration_s", "start_distance_m", "end_distance_m"} <= set(events.columns)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["duration_s"] == 7.0
    assert event["end_time_s"] - event["start_time_s"] == event["duration_s"]
    assert event["start_distance_m"] <= event["end_distance_m"]


def test_find_stopped_events_drops_short_stops():
    speeds = np.full(20, 5.0)
    # only 3 slow samples -> 2s < MIN_STOPPED_RUN_SECONDS, should be dropped
    speeds[5:8] = 0.0
    traj = _trajectory(np.linspace(0, 200, 20))
    events = find_stopped_events(traj, pd.Series(speeds, index=traj.index))
    assert events.empty


def test_find_stopped_events_no_slow_samples_is_empty():
    traj = _trajectory(np.linspace(0, 200, 20))
    fast = pd.Series(np.full(20, STOPPED_SPEED_THRESHOLD_MPS + 1.0), index=traj.index)
    events = find_stopped_events(traj, fast)
    assert events.empty
    assert "end_distance_m" in events.columns  # columns preserved even when empty


def test_identify_nearside_signals_flags_signals_near_a_nearside_stop():
    signals = pd.Series([100.0, 500.0, 900.0], index=[0, 1, 2])
    stops = pd.Series([110.0, 700.0], index=["a", "b"])
    nearside = pd.Series([True, False], index=["a", "b"])

    flags = identify_nearside_signals(signals, stops, nearside, nearside_stop_distance=20)

    assert flags.loc[0]  # signal 0 is 10 m from near-side stop "a"
    assert not flags.loc[1]
    assert not flags.loc[2]  # near stop "b" but "b" is not near-side


def test_identify_nearside_signals_no_nearside_stops_returns_all_false():
    signals = pd.Series([100.0, 500.0], index=[0, 1])
    stops = pd.Series([110.0], index=["a"])
    nearside = pd.Series([False], index=["a"])
    flags = identify_nearside_signals(signals, stops, nearside, nearside_stop_distance=20)
    assert not flags.any()
