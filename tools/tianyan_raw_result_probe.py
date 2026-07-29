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

"""Direct cqlib-tianyan protocol probe with no VQE-layer interpretation.

This script intentionally performs no error recovery. Any login, submission,
result-order, result-header, or result-format problem terminates with the
original traceback or an assertion failure.
"""

from __future__ import annotations

import argparse
import os
import re

from cqlib_tianyan import TianyanPlatform

_MEASURE = re.compile(r"(?m)^M Q(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--qubits",
        required=True,
        help="One or two physical qubits, e.g. 1 or 1,8",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument(
        "--api-key",
        default=os.getenv("TIANYAN_API_KEY"),
        help="Omit to use locally saved credentials",
    )
    return parser.parse_args()


def qubit_index(qubit) -> int:
    return int(qubit.index)


def main() -> None:
    args = parse_args()
    physical = tuple(int(value) for value in args.qubits.split(","))
    assert len(physical) in (1, 2)
    assert len(set(physical)) == len(physical)

    if args.api_key:
        platform = TianyanPlatform.login(args.api_key)
    else:
        platform = TianyanPlatform.from_credentials()
    backend = platform.get_backend(args.device)

    q0 = physical[0]
    circuits = [
        f"M Q{q0}",
        f"X Q{q0}\nM Q{q0}",
    ]
    labels = ["|0>, measure Z", "|1>, measure Z"]

    if len(physical) == 2:
        q1 = physical[1]
        circuits.extend(
            [
                f"X Q{q0}\nM Q{q0}\nM Q{q1}",
                f"X Q{q0}\nM Q{q1}\nM Q{q0}",
            ]
        )
        labels.extend(
            [
                "prepare Q0=1, measurement header [Q0,Q1]",
                "prepare Q0=1, measurement header [Q1,Q0]",
            ]
        )

    print(f"backend: {backend!r}")
    print(f"circuits: {len(circuits)}")
    for index, (label, circuit) in enumerate(zip(labels, circuits, strict=True)):
        print(f"\n--- circuit[{index}] {label} ---")
        print(circuit)

    task = backend.run_with_mode(circuits, args.shots, mode="disabled")
    print(f"\ntask_ids: {task.task_ids}")
    results = task.wait(
        timeout_secs=args.timeout,
        poll_interval_secs=args.poll,
    )

    assert len(results) == len(task.task_ids) == len(circuits)

    for index, (result, task_id, circuit) in enumerate(
        zip(results, task.task_ids, circuits, strict=True)
    ):
        expected_qubits = tuple(int(value) for value in _MEASURE.findall(circuit))
        actual_qubits = tuple(qubit_index(q) for q in result.qubits)

        print(f"\n=== result[{index}] ===")
        print(f"task_id expected : {task_id}")
        print(f"task_id actual   : {result.task_id}")
        print(f"qubits expected  : {expected_qubits}")
        print(f"qubits actual    : {actual_qubits}")
        print(f"shots            : {result.shots}")
        print(f"counts           : {result.counts}")
        print(f"probabilities    : {result.probabilities}")

        assert str(result.task_id) == str(task_id)
        assert actual_qubits == expected_qubits
        assert result.counts
        assert sum(result.counts.values()) > 0


if __name__ == "__main__":
    main()
