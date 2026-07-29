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

"""Truth-table test for Tianyan X/Y/Z measurement basis rotations.

Every case uses one Pauli term and ``grouping='none'``. No grouping logic can
influence these results. Readout calibration is disabled so the SDK cannot use
its automatic fallback path.
"""

from __future__ import annotations

import argparse
import math
import os

from cqlib import Circuit
from cqlib_tianyan import TianyanPlatform

from cqlib_vqe import TianyanEnergyEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--qubit", required=True, type=int)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--api-key", default=os.getenv("TIANYAN_API_KEY"))
    return parser.parse_args()


def make_platform(api_key: str | None):
    if api_key:
        return TianyanPlatform.login(api_key)
    return TianyanPlatform.from_credentials()


def state_zero() -> Circuit:
    return Circuit(1)


def state_one() -> Circuit:
    circuit = Circuit(1)
    circuit.x(0)
    return circuit


def state_plus() -> Circuit:
    circuit = Circuit(1)
    circuit.h(0)
    return circuit


def state_plus_i() -> Circuit:
    circuit = Circuit(1)
    circuit.rx(0, -math.pi / 2)
    return circuit


def main() -> None:
    args = parse_args()
    platform = make_platform(args.api_key)
    backend = platform.get_backend(args.device)
    estimator = TianyanEnergyEstimator(
        backend,
        1,
        shots=args.shots,
        calibration_mode="disabled",
        physical_qubits=[args.qubit],
        grouping="none",
        timeout_secs=args.timeout,
        poll_interval_secs=args.poll,
    )

    cases = [
        ("|0>, <Z>", state_zero(), "Z", +1.0),
        ("|1>, <Z>", state_one(), "Z", -1.0),
        ("|+>, <X>", state_plus(), "X", +1.0),
        ("|+i>, <Y>", state_plus_i(), "Y", +1.0),
        ("|0>, <X>", state_zero(), "X", 0.0),
        ("|0>, <Y>", state_zero(), "Y", 0.0),
    ]

    failures: list[str] = []
    for label, circuit, pauli, expected in cases:
        plan = estimator.prepare(circuit, [(pauli, 1.0)])
        assert len(plan.circuits) == 1
        print(f"\n--- {label} ---")
        print(plan.circuits[0])

        result = estimator.evaluate_with_details(circuit, [(pauli, 1.0)])
        measured = result.groups[0].terms[0].expectation
        error = abs(measured - expected)
        print(f"expected={expected:+.6f} measured={measured:+.6f} error={error:.6f}")
        if error > args.tolerance:
            failures.append(
                f"{label}: expected {expected:+.6f}, measured {measured:+.6f}"
            )

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
