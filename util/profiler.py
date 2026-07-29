"""Backward-compatible import path for the profiler."""
from utils.profiler import PerformanceTracker, global_tracker

__all__ = ["PerformanceTracker", "global_tracker"]
