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

"""Microbenchmark native cqlib2 Pauli rotations against gate decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from cqlib.qis import PauliString, Statevector


def apply_decomposed(state: Statevector, pauli: str, theta: float) -> None:
    active = [(q, char) for q, char in enumerate(pauli) if char != "I"]
    for q, char in active:
        if char == "X":
            state.apply_h(q)
        elif char == "Y":
            state.apply_rx(q, np.pi / 2)
    qubits = tuple(q for q, _ in active)
    chain = tuple(zip(qubits[:-1], qubits[1:]))
    for c, t in chain:
        state.apply_cx(c, t)
    if qubits:
        state.apply_rz(qubits[-1], theta)
    for c, t in reversed(chain):
        state.apply_cx(c, t)
    for q, char in active:
        if char == "X":
            state.apply_h(q)
        elif char == "Y":
            state.apply_rx(q, -np.pi / 2)


def workload(n_qubits: int, terms: int, seed: int):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(terms):
        support = rng.choice(n_qubits, size=min(4, n_qubits), replace=False)
        chars = ["I"] * n_qubits
        for q in support:
            chars[int(q)] = str(rng.choice(list("XYZ")))
        pauli = "".join(chars)
        records.append((pauli, PauliString.from_str(pauli[::-1]), float(rng.normal(scale=0.1))))
    return records


def run_native(n_qubits, records):
    state = Statevector(n_qubits)
    for pauli, native, theta in records:
        state.apply_pauli_rotation(native, theta)
    return float(np.sum(np.abs(np.asarray(state.data))))


def run_decomposed(n_qubits, records):
    state = Statevector(n_qubits)
    for pauli, _, theta in records:
        apply_decomposed(state, pauli, theta)
    return float(np.sum(np.abs(np.asarray(state.data))))


def time_call(fn, repeats):
    values = []
    checksum = None
    for _ in range(repeats):
        start = perf_counter()
        checksum = fn()
        values.append(perf_counter() - start)
    return values, checksum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--terms", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not callable(getattr(Statevector, "apply_pauli_rotation", None)):
        raise RuntimeError("native Pauli rotation is not installed")
    records = workload(args.qubits, args.terms, args.seed)
    run_native(args.qubits, records[:5])
    run_decomposed(args.qubits, records[:5])
    native_times, native_checksum = time_call(
        lambda: run_native(args.qubits, records), args.repeats
    )
    decomposed_times, decomposed_checksum = time_call(
        lambda: run_decomposed(args.qubits, records), args.repeats
    )
    native_median = median(native_times)
    decomposed_median = median(decomposed_times)
    payload = {
        "qubits": args.qubits,
        "terms": args.terms,
        "repeats": args.repeats,
        "native_seconds": native_times,
        "decomposed_seconds": decomposed_times,
        "native_median_seconds": native_median,
        "decomposed_median_seconds": decomposed_median,
        "speedup": decomposed_median / native_median,
        "checksums": [native_checksum, decomposed_checksum],
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
