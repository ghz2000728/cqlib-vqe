"""Backward-compatible alias for :mod:`utils`."""
from utils.profiler import PerformanceTracker, global_tracker

__all__ = ["PerformanceTracker", "global_tracker"]
