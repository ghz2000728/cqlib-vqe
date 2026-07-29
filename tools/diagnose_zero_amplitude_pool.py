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

"""Diagnose zero-CCSD-amplitude generators in the full LiH canonical pool."""

from __future__ import annotations

import argparse
import math

import numpy as np

from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine
from cqlib_vqe import AdaptiveSelectedUCCSDSolver, AdaptiveSelectionConfig
from cqlib_vqe.vqe.estimator import DirectStatevectorEstimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond", type=float, default=1.596)
    parser.add_argument("--initial-threshold", type=float, default=1e-3)
    parser.add_argument("--gradient-delta", type=float, default=1e-3)
    parser.add_argument("--gradient-threshold", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--rhobeg", type=float, default=0.03)
    parser.add_argument("--focus-index", type=int, default=28)
    parser.add_argument("--trotter-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--trotter-steps", type=int, default=2)
    args = parser.parse_args()

    molecule = MolecularDataEngine(
        geometry=[
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, args.bond)),
        ],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        mapper_type="jw",
        excitation_threshold=args.initial_threshold,
        active_space=ActiveSpaceConfig.full(),
    ).run(run_fci=True, audit_ccsd_basis=True)

    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
    config = AdaptiveSelectionConfig(
        initial_amplitude_threshold=args.initial_threshold,
        candidate_amplitude_threshold=0.0,
        pool_rounds=1,
        max_add_per_round=1,
        gradient_delta=args.gradient_delta,
        gradient_threshold=args.gradient_threshold,
        optimizer_maxiter=args.maxiter,
        optimizer_options={"rhobeg": args.rhobeg, "catol": 1e-8},
        trotter_order=args.trotter_order,
        trotter_steps=args.trotter_steps,
    )
    solver = AdaptiveSelectedUCCSDSolver(molecule, estimator, config)

    full_pool = solver._pool()
    initial = solver._initial_selection(full_pool)
    initial_values = np.asarray([solver._amplitudes[index] for index in initial])

    print("=== pool partition before optimization ===")
    print("full_pool_indices      =", full_pool)
    print("initial_selected       =", initial)
    print("candidate_indices      =", solver._remaining_candidates(initial))
    print("full parameter count   =", len(full_pool))
    print("initial parameter count=", len(initial))
    print("focus amplitude        =", solver._amplitudes[args.focus_index])

    best = math.inf

    def callback(_params, energy: float, evaluation: int) -> None:
        nonlocal best
        best = min(best, energy)
        if evaluation == 1 or evaluation % 20 == 0:
            print(
                f"optimizer eval={evaluation:4d} current={energy:+.12f} "
                f"best={best:+.12f}"
            )

    optimization = solver._optimize(initial, initial_values, callback)
    optimized = np.asarray(optimization["optimal_params"], dtype=float)
    print("\n=== optimized seed ===")
    print("energy       =", float(optimization["optimal_value"]))
    print("FCI energy   =", molecule.fci_energy)
    print("FCI error Ha =", float(optimization["optimal_value"] - molecule.fci_energy))
    print("parameters   =", optimized.tolist())

    candidates = solver._screen(initial, optimized)
    zero_candidates = [item for item in candidates if item.ccsd_amplitude == 0.0]

    print("\n=== top complete-pool gradients ===")
    for item in candidates[: min(20, len(candidates))]:
        print(
            f"packed={item.packed_index:3d} "
            f"ccsd={item.ccsd_amplitude:+.12e} "
            f"gradient={item.gradient:+.12e} "
            f"abs={item.absolute_gradient:.12e}"
        )

    print("\n=== exact-zero CCSD-amplitude candidates ===")
    for item in zero_candidates:
        print(
            f"packed={item.packed_index:3d} "
            f"gradient={item.gradient:+.12e} "
            f"abs={item.absolute_gradient:.12e}"
        )

    focus = next(
        (item for item in candidates if item.packed_index == args.focus_index),
        None,
    )
    if focus is None:
        raise AssertionError(
            f"focus packed index {args.focus_index} is missing from the candidate pool"
        )
    print("\n=== focus candidate ===")
    print(focus)


if __name__ == "__main__":
    main()
