import random

import numpy as np
import pandas as pd

from panel360.timeseries import (
    apply_pattern,
    create_sliding_windows_from_segments,
    process_and_fill_segment,
    split_into_segments_with_gaps,
)


def test_process_and_fill_segment_interpolates_short_gap():
    index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"])
    df = pd.DataFrame({"a": [1.0, 3.0, 5.0]}, index=index)

    filled = process_and_fill_segment(df, max_gap_hours=3)

    assert len(filled) == 4
    assert filled.loc["2024-01-01 02:00", "a"] == 4.0


def test_split_into_segments_with_gaps_single_segment():
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    df = pd.DataFrame({"a": range(5)}, index=index)

    segments = split_into_segments_with_gaps(df, max_small_gap_hours=3)

    assert len(segments) == 1
    assert len(segments[0]) == 5


def test_split_into_segments_with_gaps_splits_on_large_gap():
    first = pd.date_range("2024-01-01 00:00", periods=3, freq="h")
    second = pd.date_range("2024-01-02 00:00", periods=3, freq="h")
    index = first.append(second)
    df = pd.DataFrame({"a": range(6)}, index=index)

    segments = split_into_segments_with_gaps(df, max_small_gap_hours=3)

    assert len(segments) == 2
    assert len(segments[0]) == 3
    assert len(segments[1]) == 3


def test_create_sliding_windows_from_segments_shapes():
    index = pd.date_range("2024-01-01", periods=27, freq="h")
    segment = pd.DataFrame({"a": np.arange(27)}, index=index)

    X, y = create_sliding_windows_from_segments([segment], input_days=1, forecast_horizon=2)

    assert X.shape == (2, 24, 1)
    assert y.shape == (2, 2, 1)
    np.testing.assert_array_equal(X[0].ravel(), np.arange(0, 24))
    np.testing.assert_array_equal(y[0].ravel(), np.arange(24, 26))
    np.testing.assert_array_equal(X[1].ravel(), np.arange(1, 25))
    np.testing.assert_array_equal(y[1].ravel(), np.arange(25, 27))


def test_create_sliding_windows_skips_too_short_segments():
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    segment = pd.DataFrame({"a": np.arange(10)}, index=index)

    X, y = create_sliding_windows_from_segments([segment], input_days=1, forecast_horizon=2)

    assert X.shape == (0, 0, 0)
    assert y.shape == (0, 0, 0)


def test_apply_pattern_all_6_days_zeroes_first_144_rows():
    x_seq = np.ones((168, 2))

    result = apply_pattern(x_seq, "all_6_days")

    np.testing.assert_array_equal(result[:144], 0)
    np.testing.assert_array_equal(result[144:], 1)


def test_apply_pattern_one_day_zeroes_exactly_one_day():
    x_seq = np.ones((168, 2))

    result = apply_pattern(x_seq, "one_day", rng=random.Random(0))

    zeroed_rows = np.where(~result.any(axis=1))[0]
    assert len(zeroed_rows) == 24
    assert list(zeroed_rows) == list(range(zeroed_rows[0], zeroed_rows[0] + 24))


def test_apply_pattern_two_days_zeroes_exactly_two_days():
    x_seq = np.ones((168, 2))

    result = apply_pattern(x_seq, "two_days", rng=random.Random(0))

    zeroed_rows = np.where(~result.any(axis=1))[0]
    assert len(zeroed_rows) == 48


def test_apply_pattern_unknown_pattern_is_a_noop():
    x_seq = np.ones((168, 2))

    result = apply_pattern(x_seq, "none")

    np.testing.assert_array_equal(result, x_seq)
