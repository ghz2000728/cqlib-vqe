"""Run H2 UCCSD-VQE on a Tianyan backend with grouped Pauli measurements.

This example performs real hardware submissions and may consume platform
credits.  Start with a small ``TIANYAN_MAXITER`` and ``TIANYAN_SHOTS``.

Required:
    TIANYAN_DEVICE

Optional:
    TIANYAN_API_KEY          omit to reuse saved credentials
    TIANYAN_PHYSICAL_QUBITS  comma-separated logical->physical map
    TIANYAN_SHOTS            default 2000
    TIANYAN_MAXITER          default 10
    TIANYAN_CALIBRATION      enabled/disabled, default disabled
"""

from __future__ import annotations

import os

from cqlib_vqe import MolecularDataEngine
from cqlib_vqe import TianyanEnergyEstimator, UCCSDFactory, VQESolver


def _physical_mapping(n_qubits: int) -> list[int]:
    value = os.getenv("TIANYAN_PHYSICAL_QUBITS")
    if not value:
        return list(range(n_qubits))
    mapping = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(mapping) != n_qubits:
        raise ValueError(
            f"TIANYAN_PHYSICAL_QUBITS must contain {n_qubits} entries, got {len(mapping)}"
        )
    return mapping


def main() -> None:
    device_name = os.environ["TIANYAN_DEVICE"]
    api_key = os.getenv("TIANYAN_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("TIANYAN_API_KEY is empty")
    shots = int(os.getenv("TIANYAN_SHOTS", "2000"))
    max_iter = int(os.getenv("TIANYAN_MAXITER", "10"))
    calibration_mode = os.getenv("TIANYAN_CALIBRATION", "disabled")

    molecule = MolecularDataEngine(
        geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        basis="sto-3g",
        mapper_type="jw",
    ).run()

    factory = UCCSDFactory(
        molecule,
        construction_mode="jit",
        trotter_steps=1,
        trotter_order=2,
    )
    estimator = TianyanEnergyEstimator.from_credentials(
        device_name=device_name,
        n_qubits=molecule.n_qubits,
        api_key=api_key,
        shots=shots,
        calibration_mode=calibration_mode,
        physical_qubits=_physical_mapping(molecule.n_qubits),
        measure_only_support=True,
        timeout_secs=1800.0,
        poll_interval_secs=5.0,
    )
    solver = VQESolver(
        factory,
        estimator,
        optimizer_method="COBYLA",
        max_iter=max_iter,
        execution_mode="circuit",
        optimizer_options={"rhobeg": 0.1},
    )

    def callback(parameters, energy, evaluation):
        groups = estimator.last_plan.num_measurement_circuits if estimator.last_plan else 0
        print(
            f"eval={evaluation:03d} energy={energy:+.12f} "
            f"groups={groups} tasks={len(estimator.last_result.task_ids)}"
        )

    result = solver.run(molecule.hamiltonian_data, callback=callback)
    print("\n=== Tianyan H2 VQE result ===")
    print(f"energy       : {result['optimal_value']:.12f} Ha")
    print(f"evaluations  : {result['n_evals']}")
    print(f"success      : {result['success']}")
    print(f"parameters   : {result['optimal_params']}")


if __name__ == "__main__":
    main()
