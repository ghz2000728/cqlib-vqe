"""Microbenchmark for cqlib2 VQE execution paths without OpenFermion/PySCF."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from cqlib_vqe import DirectStatevectorEstimator, NativeStatevectorEstimator, UCCSDFactory


def make_problem(num_qubits: int, num_params: int):
    generators = []
    alphabet = "XYZ"
    for index in range(num_params):
        chars = ["I"] * num_qubits
        q0 = index % num_qubits
        q1 = (index * 3 + 1) % num_qubits
        chars[q0] = alphabet[index % 3]
        chars[q1] = alphabet[(index + 1) % 3]
        generators.append([("".join(chars), 0.125)])
    hamiltonian = []
    for qubit in range(num_qubits):
        chars = ["I"] * num_qubits
        chars[qubit] = "Z"
        hamiltonian.append(("".join(chars), -1.0 / num_qubits))
    return generators, hamiltonian


def time_mode(name, iterations, fn):
    for _ in range(3):
        fn()
    start = perf_counter()
    value = None
    for _ in range(iterations):
        value = fn()
    elapsed = perf_counter() - start
    print(
        f"{name:<20} total={elapsed:9.4f}s  mean={elapsed * 1e6 / iterations:10.1f}us  "
        f"last_energy={value:.12f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--params", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    generators, hamiltonian = make_problem(args.qubits, args.params)
    params = np.linspace(-0.2, 0.2, args.params)

    jit_factory = UCCSDFactory.from_compiled(
        n_qubits=args.qubits,
        n_electrons=min(2, args.qubits),
        generators=generators,
        construction_mode="jit",
    )
    bind_factory = UCCSDFactory.from_compiled(
        n_qubits=args.qubits,
        n_electrons=min(2, args.qubits),
        generators=generators,
        construction_mode="bind",
    )
    native = NativeStatevectorEstimator(n_qubits=args.qubits)
    direct = DirectStatevectorEstimator(n_qubits=args.qubits)

    time_mode(
        "jit+circuit",
        args.iterations,
        lambda: native.evaluate(jit_factory.build(params), hamiltonian),
    )
    time_mode(
        "bind+circuit",
        args.iterations,
        lambda: native.evaluate(bind_factory.build(params), hamiltonian),
    )
    time_mode(
        "direct_statevector",
        args.iterations,
        lambda: direct.evaluate_parameters(jit_factory, params, hamiltonian),
    )


if __name__ == "__main__":
    main()
