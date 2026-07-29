#!/usr/bin/env python3
"""Expose the true local energy geometry around the CCSD UCC initial point."""

from __future__ import annotations

import argparse
import numpy as np

from examples.run_local_accuracy import MOLECULES
from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine
from cqlib_vqe import DirectStatevectorEstimator, UCCSDFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", choices=MOLECULES)
    parser.add_argument("--active-space", choices=["default", "full", "auto"], default="default")
    parser.add_argument("--max-active-qubits", type=int)
    parser.add_argument("--pool-mode", choices=["selected", "full"], default="selected")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--step", type=float, default=0.03)
    parser.add_argument("--trotter-steps", type=int, default=2)
    args = parser.parse_args()

    geometry, default_budget = MOLECULES[args.molecule]
    if args.active_space == "full" or (args.active_space == "default" and default_budget is None):
        active = ActiveSpaceConfig.full()
    else:
        budget = args.max_active_qubits or default_budget
        if budget is None:
            raise ValueError("--max-active-qubits is required")
        active = ActiveSpaceConfig.automatic(max_active_qubits=budget, freeze_core=True)

    molecule = MolecularDataEngine(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        mapper_type="jw",
        excitation_threshold=args.threshold,
        active_space=active,
    ).run(run_fci=True, audit_ccsd_basis=True)

    indices = None
    if args.pool_mode == "full":
        indices = [item.packed_index for item in molecule.ccsd_descriptors]
    factory = UCCSDFactory(
        molecule,
        construction_mode="bind",
        trotter_steps=args.trotter_steps,
        trotter_order=2,
        selected_packed_indices=indices,
    )
    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)

    def energy(values) -> float:
        return float(estimator.evaluate_parameters(factory, values, molecule.hamiltonian_data))

    x0 = np.asarray(factory.initial_values, dtype=float)
    zero = np.zeros_like(x0)
    print(f"molecule       : {args.molecule}")
    print(f"pool           : {factory.num_params}/{factory.full_parameter_count}")
    print(f"packed indices : {factory.selected_packed_indices}")
    print(f"CCSD amplitudes: {x0.tolist()}")
    print(f"E(HF / zero)   : {energy(zero):+.12f}")
    print(f"E(+CCSD)       : {energy(x0):+.12f}")
    print(f"E(-CCSD)       : {energy(-x0):+.12f}")
    print(f"E(FCI)         : {molecule.fci_energy:+.12f}")

    print("\nscale scan E(alpha * CCSD):")
    for scale in (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        print(f"  alpha={scale:+.2f} E={energy(scale * x0):+.12f}")

    delta = args.delta
    gradient = np.zeros_like(x0)
    print("\ncentral finite-difference gradient at +CCSD:")
    for index in range(x0.size):
        plus = x0.copy(); plus[index] += delta
        minus = x0.copy(); minus[index] -= delta
        e_plus = energy(plus)
        e_minus = energy(minus)
        gradient[index] = (e_plus - e_minus) / (2.0 * delta)
        print(
            f"  p={index:3d} packed={factory.selected_packed_indices[index]:3d} "
            f"theta={x0[index]:+.6e} grad={gradient[index]:+.6e} "
            f"E-={e_minus:+.12f} E+={e_plus:+.12f}"
        )

    norm = float(np.linalg.norm(gradient))
    print(f"\n||gradient|| = {norm:.6e}")
    if norm > np.finfo(float).eps:
        direction = -gradient / norm
        print("descent-direction line scan:")
        base = energy(x0)
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
            step = args.step * factor
            trial = energy(x0 + step * direction)
            print(
                f"  step={step:.6e} E={trial:+.12f} "
                f"deltaE={trial - base:+.6e}"
            )


if __name__ == "__main__":
    main()
