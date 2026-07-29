#!/usr/bin/env python3
"""Numerically validate chemistry preprocessing against PySCF references.

This script is intentionally fail-fast.  It checks that the Hartree--Fock
Slater determinant evaluated with the mapped active-space Hamiltonian equals
the PySCF HF energy.  This catches missing frozen-core constants, incorrect
active-orbital ordering, Pauli endianness mistakes, and reference-state errors.
"""

from __future__ import annotations

import argparse

from cqlib import Circuit

from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine
from cqlib_vqe import NativeStatevectorEstimator, UCCSDFactory


CASES = {
    "h2": {
        "geometry": [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        "active": ActiveSpaceConfig.full(),
    },
    "lih": {
        "geometry": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.596))],
        "active": ActiveSpaceConfig.automatic(max_active_qubits=10, freeze_core=True),
    },
    "h2o": {
        "geometry": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.587)),
            ("H", (0.0, -0.757, 0.587)),
        ],
        "active": ActiveSpaceConfig.automatic(max_active_qubits=12, freeze_core=True),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", choices=[*CASES, "all"], default="all", nargs="?")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--hf-tolerance", type=float, default=1e-8)
    parser.add_argument("--run-fci", action="store_true")
    parser.add_argument("--disable-ccsd-audit", action="store_true")
    args = parser.parse_args()

    selected = list(CASES) if args.molecule == "all" else [args.molecule]
    for name in selected:
        case = CASES[name]
        print(f"\n{'=' * 24} {name.upper()} {'=' * 24}")
        molecule = MolecularDataEngine(
            geometry=case["geometry"],
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            mapper_type="jw",
            excitation_threshold=args.threshold,
            active_space=case["active"],
        ).run(
            run_fci=args.run_fci,
            audit_ccsd_basis=not args.disable_ccsd_audit,
        )

        reference = Circuit(molecule.n_qubits)
        for qubit in range(molecule.n_electrons):
            reference.x(qubit)
        estimator = NativeStatevectorEstimator(n_qubits=molecule.n_qubits)
        mapped_hf = estimator.evaluate(reference, molecule.hamiltonian_data)
        delta = mapped_hf - molecule.hf_energy

        factory = UCCSDFactory(
            molecule,
            construction_mode="jit",
            trotter_steps=1,
            trotter_order=2,
        )
        report = molecule.active_space_report
        print(f"active-space report : {report.to_dict() if report else None}")
        print(f"mapped HF energy    : {mapped_hf:+.12f} Ha")
        print(f"PySCF HF energy     : {molecule.hf_energy:+.12f} Ha")
        print(f"HF difference       : {delta:+.3e} Ha")
        print(f"active CCSD energy  : {molecule.ccsd_energy:+.12f} Ha")
        print(
            "canonical pool      : "
            f"{factory.num_params}/{factory.full_parameter_count} selected"
        )
        print(f"packed indices      : {factory.selected_packed_indices}")
        print(f"noncommuting indices: {factory.noncommuting_packed_indices}")
        if abs(delta) > args.hf_tolerance:
            raise AssertionError(
                f"{name}: mapped HF differs from PySCF by {delta:+.3e} Ha"
            )
        print("correctness check   : PASS")


if __name__ == "__main__":
    main()
