"""Gap-aware segmentation, resampling, and windowing of latent-vector time series."""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def process_and_fill_segment(segment, max_gap_hours=3):
    """Resample a segment to hourly frequency and linearly interpolate short gaps."""
    segment = segment.resample("h").asfreq()
    segment = segment.interpolate(method="linear", limit=max_gap_hours, limit_direction="forward")
    segment.dropna(how="any", inplace=True)
    return segment


def split_into_segments_with_gaps(df, max_small_gap_hours=3):
    """Split a time-indexed DataFrame into continuous, hourly-filled segments.

    Consecutive rows more than ``max_small_gap_hours`` apart start a new
    segment; each segment is resampled/interpolated independently. If no
    such gap exists, the whole dataset is treated as one segment.
    """
    df = df.sort_index()
    segments = []
    start_idx = 0
    df_index = df.index
    found_large_gap = False

    for i in range(1, len(df_index)):
        time_diff_hours = (df_index[i] - df_index[i - 1]).total_seconds() / 3600.0
        if time_diff_hours > max_small_gap_hours:
            found_large_gap = True
            segment = process_and_fill_segment(df.iloc[start_idx:i].copy(), max_small_gap_hours)
            logger.info(
                "Segment %d from %s to %s: %d rows",
                len(segments) + 1, df_index[start_idx], df_index[i - 1], len(segment),
            )
            if len(segment) > 0:
                segments.append(segment)
            start_idx = i

    if not found_large_gap:
        logger.info("No large gaps found. Treating entire dataset as a single segment.")
        segment = process_and_fill_segment(df.copy(), max_small_gap_hours)
        if len(segment) > 0:
            segments.append(segment)
    elif start_idx < len(df):
        segment = process_and_fill_segment(df.iloc[start_idx:].copy(), max_small_gap_hours)
        logger.info(
            "Final segment %d from %s to %s: %d rows",
            len(segments) + 1, df_index[start_idx], df_index[-1], len(segment),
        )
        if len(segment) > 0:
            segments.append(segment)

    logger.info("Total segments created: %d", len(segments))
    return segments


def create_sliding_windows_from_segments(segments, input_days=7, forecast_horizon=24):
    """Build sliding ``(X, y)`` windows from each segment and concatenate them."""
    sequence_length = input_days * 24
    all_X, all_y = [], []

    for seg in segments:
        if len(seg) < (sequence_length + forecast_horizon):
            continue
        data = seg.values
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            all_X.append(data[i:i + sequence_length])
            all_y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])

    if not all_X:
        return np.empty((0, 0, 0)), np.empty((0, 0, 0))

    return np.array(all_X), np.array(all_y)


def apply_pattern(x_seq, pattern, rng=random):
    """Zero out part of a 7-day (168-hour) input sequence to simulate data gaps.

    ``pattern`` is one of ``all_6_days`` (zero the first 6 of 7 days),
    ``one_day`` (zero one random day among the first 6), ``two_days``
    (zero two random days among the first 6), or anything else (no-op,
    e.g. ``none``).
    """
    x_modified = x_seq.copy()
    if pattern == "all_6_days":
        x_modified[:144, :] = 0
    elif pattern == "one_day":
        day_to_blank = rng.randint(0, 5)
        start, end = day_to_blank * 24, (day_to_blank + 1) * 24
        x_modified[start:end, :] = 0
    elif pattern == "two_days":
        for day_to_blank in rng.sample(range(6), 2):
            start, end = day_to_blank * 24, (day_to_blank + 1) * 24
            x_modified[start:end, :] = 0
    return x_modified
