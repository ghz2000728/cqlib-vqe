#!/usr/bin/env python3
"""Check that the chemistry, factory, and adaptive modules are one compatible source set."""
from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path

import chemistry
import vqe
import cqlib_vqe
from cqlib_vqe import AdaptiveSelectionConfig, AdaptiveSelectedUCCSDSolver, UCCSDFactory

root = Path(__file__).resolve().parents[1]
paths = {
    "chemistry": Path(chemistry.__file__).resolve(),
    "vqe": Path(vqe.__file__).resolve(),
    "factory": Path(inspect.getsourcefile(UCCSDFactory) or "").resolve(),
    "adaptive_solver": Path(inspect.getsourcefile(AdaptiveSelectedUCCSDSolver) or "").resolve(),
}

print("project root   :", root)
print("canonical import:", Path(cqlib_vqe.__file__).resolve())
print("package version:", importlib.metadata.version("cqlib-vqe"))
for name, path in paths.items():
    print(f"{name:15s}: {path}")
    path.relative_to(root)

factory_params = inspect.signature(UCCSDFactory.__init__).parameters
for required in ("selected_packed_indices", "trotter_order", "trotter_steps"):
    if required not in factory_params:
        raise RuntimeError(f"UCCSDFactory is missing {required!r}")

adaptive_params = inspect.signature(AdaptiveSelectionConfig).parameters
for required in (
    "initial_amplitude_threshold",
    "candidate_amplitude_threshold",
    "pool_rounds",
    "gradient_threshold",
    "max_add_per_round",
    "max_screen_candidates",
):
    if required not in adaptive_params:
        raise RuntimeError(f"AdaptiveSelectionConfig is missing {required!r}")

print("source synchronization: PASS")
