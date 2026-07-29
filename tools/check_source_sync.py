#!/usr/bin/env python3
# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check that all public modules resolve to the single cqlib_vqe source tree."""
from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path

import cqlib_vqe
from cqlib_vqe import chemistry, vqe
from cqlib_vqe import AdaptiveSelectionConfig, AdaptiveSelectedUCCSDSolver, UCCSDFactory

root = Path(__file__).resolve().parents[1]
paths = {
    "chemistry": Path(chemistry.__file__).resolve(),
    "vqe": Path(vqe.__file__).resolve(),
    "factory": Path(inspect.getsourcefile(UCCSDFactory) or "").resolve(),
    "adaptive_solver": Path(inspect.getsourcefile(AdaptiveSelectedUCCSDSolver) or "").resolve(),
}

print("project root   :", root)
print("package import  :", Path(cqlib_vqe.__file__).resolve())
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

print("single source tree: PASS")
