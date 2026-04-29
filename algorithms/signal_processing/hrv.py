from __future__ import annotations

from typing import Sequence


def _generate_mock_rr_intervals(num_beats: int = 12) -> list[int]:
    """Generate a small deterministic RR-interval sequence for placeholder HRV output."""
    base_pattern = [810, 790, 830, 805, 815, 800, 825, 795, 820, 810, 805, 815]
    if num_beats <= len(base_pattern):
        return base_pattern[:num_beats]

    intervals = list(base_pattern)
    while len(intervals) < num_beats:
        intervals.extend(base_pattern)
    return intervals[:num_beats]


def calculate_hrv(rr_intervals: Sequence[int] | None = None) -> dict[str, object]:
    """Return a mock HRV result without running real variability analysis."""
    intervals = list(rr_intervals) if rr_intervals else _generate_mock_rr_intervals()

    return {
        "module": "hrv",
        "status": "placeholder",
        "algorithm": "HRV statistics reserved, not enabled",
        "summary": "HRV placeholder calculation completed. A moderate recovery-like mock result was returned for integration.",
        "risk_level": "low",
        "metrics": {
            "mean_rr_ms": 810,
            "sdnn_ms": 42,
            "rmssd_ms": 35,
            "pnn50": 0.18,
            "stress_level": "moderate",
        },
        "visualization": {
            "rr_intervals_ms": intervals,
        },
        "meta": {
            "input_provided": bool(rr_intervals),
            "placeholder_only": True,
        },
    }


def analyze_hrv(rr_intervals: Sequence[int] | None = None) -> dict[str, object]:
    """Alias kept for convenient agent-side calling."""
    return calculate_hrv(rr_intervals=rr_intervals)


__all__ = ["calculate_hrv", "analyze_hrv"]
