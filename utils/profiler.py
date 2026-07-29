"""Lightweight phase profiler used by the VQE hot path."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class PerformanceTracker:
    def __init__(self) -> None:
        self.records: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.records[name] += perf_counter() - start
            self.counts[name] += 1

    def reset(self) -> None:
        self.records.clear()
        self.counts.clear()

    def report(self) -> None:
        print("\n" + "=" * 72)
        print(f"{'Phase':<36} {'Calls':>8} {'Total (s)':>12} {'Average (ms)':>14}")
        print("-" * 72)
        for name, duration in sorted(
            self.records.items(), key=lambda item: item[1], reverse=True
        ):
            count = self.counts[name]
            average_ms = duration * 1000.0 / count if count else 0.0
            print(f"{name:<36} {count:>8d} {duration:>12.6f} {average_ms:>14.3f}")
        print("-" * 72)
        print(f"{'Total tracked (nested phases included)':<36} {'':>8} {sum(self.records.values()):>12.6f}")
        print("=" * 72 + "\n")


global_tracker = PerformanceTracker()
