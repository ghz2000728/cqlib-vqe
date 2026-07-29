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

"""No-OpenFermion smoke test using a precompiled one-parameter ansatz."""

from cqlib_vqe import DirectStatevectorEstimator, UCCSDFactory, VQESolver


def main() -> None:
    # G = i * 0.5 * Y, so angle = -theta.
    factory = UCCSDFactory.from_compiled(
        n_qubits=1,
        n_electrons=0,
        generators=[[("Y", 0.5)]],
        initial_values=[0.1],
    )
    estimator = DirectStatevectorEstimator(n_qubits=1)
    solver = VQESolver(factory, estimator, max_iter=30, execution_mode="auto")
    result = solver.run([("Z", 1.0)])
    print(result)


if __name__ == "__main__":
    main()
