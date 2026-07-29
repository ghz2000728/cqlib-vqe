"""Compare ungrouped and QWC-grouped energy measurements on the same state.

The |++> state has <XI>=<IX>=<XX>=1 and zero expectation for Z-containing
non-identity terms. The ungrouped run is the hardware reference; the grouped
run must agree within sampling noise.
"""

from __future__ import annotations

import argparse
import os

from cqlib import Circuit
from cqlib_tianyan import TianyanPlatform

from cqlib_vqe import TianyanEnergyEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--qubits", required=True, help="Two physical qubits, e.g. 1,8"
    )
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--api-key", default=os.getenv("TIANYAN_API_KEY"))
    return parser.parse_args()


def make_platform(api_key: str | None):
    if api_key:
        return TianyanPlatform.login(api_key)
    return TianyanPlatform.from_credentials()


def print_result(name, result) -> None:
    print(f"\n=== {name} ===")
    print(f"energy: {result.energy:+.10f}")
    print(f"tasks : {len(result.task_ids)} {list(result.task_ids)}")
    for group in result.groups:
        print(
            f"group {group.master_pauli} qubits={group.measured_physical_qubits} "
            f"contribution={group.contribution:+.10f}"
        )
        for term in group.terms:
            print(
                f"  {term.pauli}: expectation={term.expectation:+.10f} "
                f"coefficient={term.coefficient:+.10f} "
                f"contribution={term.contribution:+.10f}"
            )


def main() -> None:
    args = parse_args()
    physical = [int(value) for value in args.qubits.split(",")]
    assert len(physical) == 2 and len(set(physical)) == 2

    platform = make_platform(args.api_key)
    backend = platform.get_backend(args.device)

    circuit = Circuit(2)
    circuit.h(0)
    circuit.h(1)
    hamiltonian = [
        ("II", 0.25),
        ("XI", 0.70),
        ("IX", -0.20),
        ("XX", 1.10),
        ("ZI", 0.30),
        ("IZ", -0.40),
        ("ZZ", 0.50),
    ]
    exact = 1.85

    common = dict(
        backend=backend,
        n_qubits=2,
        shots=args.shots,
        calibration_mode="disabled",
        physical_qubits=physical,
        timeout_secs=args.timeout,
        poll_interval_secs=args.poll,
    )
    ungrouped = TianyanEnergyEstimator(**common, grouping="none")
    grouped = TianyanEnergyEstimator(**common, grouping="qwc")

    ungrouped_plan = ungrouped.prepare(circuit, hamiltonian)
    grouped_plan = grouped.prepare(circuit, hamiltonian)
    print(f"ungrouped circuits: {ungrouped_plan.num_measurement_circuits}")
    print(f"QWC circuits      : {grouped_plan.num_measurement_circuits}")
    print("\n--- ungrouped QCIS ---")
    for i, script in enumerate(ungrouped_plan.circuits):
        print(f"\n[{i}]\n{script}")
    print("\n--- grouped QCIS ---")
    for i, script in enumerate(grouped_plan.circuits):
        print(f"\n[{i}]\n{script}")

    ungrouped_result = ungrouped.evaluate_with_details(circuit, hamiltonian)
    grouped_result = grouped.evaluate_with_details(circuit, hamiltonian)
    print_result("ungrouped", ungrouped_result)
    print_result("QWC grouped", grouped_result)

    ungrouped_error = abs(ungrouped_result.energy - exact)
    grouped_error = abs(grouped_result.energy - exact)
    difference = abs(grouped_result.energy - ungrouped_result.energy)
    print(f"\nexact theoretical energy : {exact:+.10f}")
    print(f"ungrouped absolute error : {ungrouped_error:.10f}")
    print(f"grouped absolute error   : {grouped_error:.10f}")
    print(f"grouped-vs-ungrouped     : {difference:.10f}")

    assert ungrouped_error <= args.tolerance
    assert grouped_error <= args.tolerance
    assert difference <= args.tolerance


if __name__ == "__main__":
    main()
