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

"""Verify cqlib2's optional native Pauli rotation against gate decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cqlib.qis import PauliString, Statevector


def apply_decomposed(state: Statevector, pauli_q0_left: str, theta: float) -> None:
    active = [(q, char) for q, char in enumerate(pauli_q0_left) if char != "I"]
    for qubit, char in active:
        if char == "X":
            state.apply_h(qubit)
        elif char == "Y":
            state.apply_rx(qubit, np.pi / 2)
    qubits = tuple(q for q, _ in active)
    chain = tuple(zip(qubits[:-1], qubits[1:]))
    for control, target in chain:
        state.apply_cx(control, target)
    if qubits:
        state.apply_rz(qubits[-1], theta)
    for control, target in reversed(chain):
        state.apply_cx(control, target)
    for qubit, char in active:
        if char == "X":
            state.apply_h(qubit)
        elif char == "Y":
            state.apply_rx(qubit, -np.pi / 2)


def prepare_state(n_qubits: int) -> Statevector:
    state = Statevector(n_qubits)
    for q in range(n_qubits):
        state.apply_rx(q, 0.13 * (q + 1))
    for q in range(n_qubits - 1):
        state.apply_cx(q, q + 1)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not callable(getattr(Statevector, "apply_pauli_rotation", None)):
        raise RuntimeError(
            "Installed cqlib2 does not expose Statevector.apply_pauli_rotation. "
            "Build the minimal cqlib2 patch with maturin develop --release."
        )

    cases = [
        ("Y", 0.41),
        ("XYZI", -0.27),
        ("IXYZZI", 0.63),
        ("ZZZZ", -0.18),
    ]
    records = []
    for pauli, theta in cases:
        native = prepare_state(len(pauli))
        decomposed = prepare_state(len(pauli))
        native.apply_pauli_rotation(PauliString.from_str(pauli[::-1]), theta)
        apply_decomposed(decomposed, pauli, theta)
        error = float(np.max(np.abs(np.asarray(native.data) - np.asarray(decomposed.data))))
        records.append({"pauli_q0_left": pauli, "theta": theta, "max_error": error})
        print(f"{pauli:8s} theta={theta:+.3f} max_error={error:.3e}")
        if error > 1e-11:
            raise AssertionError(f"native Pauli rotation mismatch for {pauli}: {error}")

    payload = {"status": "ok", "cases": records}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("native Pauli rotation verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
