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

"""End-to-end H2 VQE example for cqlib2-dynamic."""

from cqlib_vqe import MolecularDataEngine
from cqlib_vqe.utils.profiler import global_tracker
from cqlib_vqe import DirectStatevectorEstimator, UCCSDFactory, VQESolver


def main() -> None:
    molecule = MolecularDataEngine(
        geometry=[
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.735)),
        ],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        mapper_type="jw",
        excitation_threshold=1e-3,
    ).run()

    factory = UCCSDFactory(molecule, construction_mode="jit", trotter_order=2)
    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
    solver = VQESolver(
        factory,
        estimator,
        optimizer_method="COBYLA",
        max_iter=100,
        tol=1e-7,
        execution_mode="auto",
    )

    def callback(_params, energy: float, evaluation: int) -> None:
        if evaluation == 1 or evaluation % 10 == 0:
            print(f"eval={evaluation:4d}  energy={energy:.12f} Ha")

    result = solver.run(molecule.hamiltonian_data, callback=callback)
    print("\n=== VQE result ===")
    for key, value in result.items():
        if key != "history":
            print(f"{key}: {value}")
    print("reference energies:", molecule.reference_energies)
    global_tracker.report()


if __name__ == "__main__":
    main()
