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

from __future__ import annotations

import pytest

from cqlib import Circuit

from cqlib_vqe.vqe import NativeStatevectorEstimator


def test_prepared_hamiltonian_is_rebuilt_after_in_place_mutation():
    estimator = NativeStatevectorEstimator(n_qubits=1)
    circuit = Circuit(1)
    hamiltonian = [["Z", 1.0]]

    estimator.prepare_hamiltonian(hamiltonian)
    assert estimator.evaluate(circuit, hamiltonian) == pytest.approx(1.0)

    hamiltonian[0][1] = 2.0
    assert estimator.evaluate(circuit, hamiltonian) == pytest.approx(2.0)


def test_hamiltonian_cache_handles_equivalent_and_reordered_inputs():
    estimator = NativeStatevectorEstimator(n_qubits=1)
    circuit = Circuit(1)

    original = [("Z", 1.0), ("I", 2.0)]
    estimator.prepare_hamiltonian(original)
    assert estimator.evaluate(circuit, original) == pytest.approx(3.0)
    assert estimator.evaluate(circuit, list(original)) == pytest.approx(3.0)
    assert estimator.evaluate(circuit, list(reversed(original))) == pytest.approx(3.0)


def test_hamiltonian_cache_preserves_duplicate_terms():
    estimator = NativeStatevectorEstimator(n_qubits=1)
    circuit = Circuit(1)
    duplicated = [("Z", 1.0), ("Z", 1.0)]

    estimator.prepare_hamiltonian(duplicated)

    assert estimator.evaluate(circuit, duplicated) == pytest.approx(2.0)
