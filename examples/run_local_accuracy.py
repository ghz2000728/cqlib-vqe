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

"""Run canonical singlet-UCCSD with correctness-oriented pool growth."""

from __future__ import annotations

import argparse
import math

from cqlib import Circuit

from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine
from cqlib_vqe import (
    AdaptiveSelectedUCCSDSolver,
    AdaptiveSelectionConfig,
    DirectStatevectorEstimator,
    NativeStatevectorEstimator,
    UCCSDFactory,
    VQESolver,
)


MOLECULES = {
    "h2": ([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.735))], None),
    "h4": ([
        ('H', (0.0, 0.0, -1.5)),
        ('H', (0.0, 0.0, -0.5)),
        ('H', (0.0, 0.0, 0.5)),
        ('H', (0.0, 0.0, 1.5)),
    ], None),
    "lih": ([('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.596))], 10),
    "beh2": ([
        ('H', (0.0, 0.0, -1.3264)),
        ('Be', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.3264)),
    ], 12),
    "h2o": ([
        ('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.757, 0.587)),
        ('H', (0.0, -0.757, 0.587)),
    ], 12),
}

CHEMICAL_ACCURACY = 1.6e-3


def build_molecule(args):
    geometry, default_budget = MOLECULES[args.molecule]
    if args.active_space == "full" or (
        args.active_space == "default" and default_budget is None
    ):
        active_space = ActiveSpaceConfig.full()
    else:
        budget = args.max_active_qubits or default_budget
        if budget is None:
            raise ValueError("--max-active-qubits is required for auto mode")
        active_space = ActiveSpaceConfig.automatic(
            max_active_qubits=budget,
            freeze_core=True,
        )
    return MolecularDataEngine(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        mapper_type="jw",
        excitation_threshold=args.initial_threshold,
        active_space=active_space,
    ).run(
        run_fci=not args.skip_fci,
        audit_ccsd_basis=not args.disable_ccsd_audit,
    )


def check_hf(molecule) -> None:
    circuit = Circuit(molecule.n_qubits)
    for qubit in range(molecule.n_electrons):
        circuit.x(qubit)
    estimator = NativeStatevectorEstimator(n_qubits=molecule.n_qubits)
    mapped_hf = estimator.evaluate(circuit, molecule.hamiltonian_data)
    delta = mapped_hf - molecule.hf_energy
    print(f"mapped HF - PySCF HF = {delta:+.3e} Ha")
    if abs(delta) > 1e-8:
        raise AssertionError("Active-space Hamiltonian/reference-state consistency failed")


def run_fixed_pool(args, molecule, estimator):
    if args.pool_mode == "full":
        packed_indices = [item.packed_index for item in molecule.ccsd_descriptors]
    else:
        packed_indices = None
    factory = UCCSDFactory(
        molecule,
        construction_mode=args.construction_mode,
        trotter_steps=args.trotter_steps,
        trotter_order=args.trotter_order,
        selected_packed_indices=packed_indices,
    )
    solver = VQESolver(
        factory,
        estimator,
        optimizer_method="COBYLA",
        max_iter=args.maxiter,
        tol=args.tol,
        optimizer_options={"rhobeg": args.rhobeg, "catol": 1e-8},
        execution_mode=args.execution_mode,
        warm_start=args.warm_start,
        warm_start_delta=args.gradient_delta,
        warm_start_step=args.warm_start_step,
    )
    best = math.inf

    def callback(_parameters, energy: float, evaluation: int) -> None:
        nonlocal best
        best = min(best, energy)
        if evaluation == 1 or evaluation % 10 == 0:
            print(
                f"eval={evaluation:4d} current={energy:+.12f} Ha "
                f"best={best:+.12f} Ha"
            )

    result = solver.run(molecule.hamiltonian_data, callback=callback)
    return result, factory.selected_packed_indices, factory.full_parameter_count


def run_adaptive(args, molecule, estimator):
    config = AdaptiveSelectionConfig(
        initial_amplitude_threshold=args.initial_threshold,
        candidate_amplitude_threshold=args.candidate_threshold,
        pool_rounds=args.pool_rounds,
        max_add_per_round=args.max_add_per_round,
        max_parameters=args.max_parameters,
        max_screen_candidates=args.max_screen_candidates,
        gradient_delta=args.gradient_delta,
        gradient_threshold=args.gradient_threshold,
        target_energy_error=args.target_error,
        optimizer_maxiter=args.maxiter,
        optimizer_tolerance=args.tol,
        optimizer_options={"rhobeg": args.rhobeg, "catol": 1e-8},
        construction_mode=args.construction_mode,
        execution_mode=args.execution_mode,
        trotter_steps=args.trotter_steps,
        trotter_order=args.trotter_order,
    )
    solver = AdaptiveSelectedUCCSDSolver(molecule, estimator, config)

    round_best = math.inf

    def optimizer_callback(_parameters, energy: float, evaluation: int) -> None:
        nonlocal round_best
        round_best = min(round_best, energy)
        if evaluation == 1 or evaluation % 20 == 0:
            print(
                f"  optimizer eval={evaluation:4d} current={energy:+.12f} "
                f"round_best={round_best:+.12f}"
            )

    def round_callback(item) -> None:
        nonlocal round_best
        added = list(item.added_indices)
        print(
            f"round={item.round_index:2d} params={len(item.selected_before):2d} "
            f"energy={item.optimized_energy:+.12f} "
            f"error={item.reference_error} grad_norm={item.gradient_norm} "
            f"added={added}"
        )
        if item.candidates:
            print("  top candidate gradients:")
            for candidate in item.candidates[: min(8, len(item.candidates))]:
                print(
                    f"    packed={candidate.packed_index:3d} "
                    f"grad={candidate.gradient:+.6e} "
                    f"ccsd={candidate.ccsd_amplitude:+.6e}"
                )
        round_best = math.inf

    result = solver.run(
        round_callback=round_callback,
        optimizer_callback=optimizer_callback,
    )
    return result.to_dict(), result.selected_packed_indices, result.full_parameter_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", choices=MOLECULES)
    parser.add_argument("--active-space", choices=["default", "full", "auto"], default="default")
    parser.add_argument("--max-active-qubits", type=int)
    parser.add_argument("--pool-mode", choices=["selected", "full", "adaptive"], default="adaptive")
    parser.add_argument("--initial-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; must remain 0.0 so zero-amplitude generators stay in the full candidate pool.",
    )
    parser.add_argument("--maxiter", type=int, default=80, help="COBYLA evaluations per optimization stage")
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--rhobeg", type=float, default=0.03)
    parser.add_argument("--warm-start", choices=["none", "gradient"], default="none")
    parser.add_argument("--warm-start-step", type=float, default=0.05)
    parser.add_argument("--gradient-delta", type=float, default=1e-3)
    parser.add_argument("--gradient-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--pool-rounds",
        "--max-rounds",
        dest="pool_rounds",
        type=int,
        default=2,
        help="maximum number of complete operator-pool gradient scans",
    )
    parser.add_argument(
        "--max-add-per-round",
        "--batch-size",
        dest="max_add_per_round",
        type=int,
        default=None,
        help="optional cap; default adds every candidate above gradient threshold",
    )
    parser.add_argument("--max-parameters", type=int)
    parser.add_argument(
        "--max-screen-candidates",
        type=int,
        help="Disabled in correctness mode because partial screening cannot certify convergence.",
    )
    parser.add_argument("--target-error", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--skip-fci", action="store_true")
    parser.add_argument("--disable-ccsd-audit", action="store_true")
    parser.add_argument("--trotter-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--trotter-steps", type=int, default=2)
    parser.add_argument("--construction-mode", choices=["jit", "bind"], default="bind")
    parser.add_argument(
        "--execution-mode",
        choices=["auto", "circuit", "direct_statevector"],
        default="auto",
    )
    parser.add_argument("--require-chemical-accuracy", action="store_true")
    args = parser.parse_args()

    molecule = build_molecule(args)
    check_hf(molecule)
    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)

    if args.pool_mode == "adaptive":
        result, selected_indices, full_count = run_adaptive(args, molecule, estimator)
        energy = float(result["optimal_value"])
        success = bool(result["precision_certified"])
        message = str(result["stop_reason"])
        evaluations = int(result["total_evaluations"])
    else:
        result, selected_indices, full_count = run_fixed_pool(args, molecule, estimator)
        energy = float(result["optimal_value"])
        success = bool(result["success"])
        message = str(result["message"])
        evaluations = int(result["n_evals"])

    if not math.isfinite(energy):
        raise FloatingPointError("VQE returned a non-finite energy")
    reference = molecule.fci_energy
    error = None if reference is None else energy - float(reference)
    chemical_accuracy = error is not None and abs(error) <= CHEMICAL_ACCURACY

    print("\n=== corrected VQE result ===")
    print(f"molecule        : {args.molecule}")
    print(f"pool mode       : {args.pool_mode}")
    print(f"active space    : {molecule.active_space_report.to_dict()}")
    if args.pool_mode == "adaptive":
        print(f"full pool       : {tuple(result['full_pool_indices'])}")
        print(f"initial selected: {tuple(result['initial_selected_packed_indices'])}")
        print(f"remaining pool  : {tuple(result['remaining_candidate_packed_indices'])}")
    print(f"selected params : {len(selected_indices)}/{full_count}")
    print(f"packed indices  : {tuple(selected_indices)}")
    print(f"VQE energy      : {energy:+.12f} Ha")
    print(f"references      : {molecule.reference_energies}")
    print(f"FCI error       : {error}")
    print(f"chemical acc.   : {chemical_accuracy}")
    print(f"evaluations     : {evaluations}")
    print(f"success         : {success} ({message})")
    if args.require_chemical_accuracy and not chemical_accuracy:
        raise AssertionError(
            f"{args.molecule} missed chemical accuracy: error={error:+.6e} Ha"
        )


if __name__ == "__main__":
    main()
