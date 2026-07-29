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

from types import SimpleNamespace

import numpy as np
import pytest

from cqlib_vqe import DirectStatevectorEstimator, MolecularDataEngine
from chemistry.mapping import encoded_occupied_qubits, map_fermion_operator
from chemistry.uccsd import canonical_singlet_operator
from vqe.factory import UCCSDFactory


@pytest.mark.parametrize(
    ("mapper", "expected"),
    [
        ("jw", (0, 1)),
        ("bk", (0,)),
        ("parity", (0,)),
    ],
)
def test_reference_occupation_uses_the_selected_binary_code(mapper, expected):
    assert encoded_occupied_qubits(
        n_spin_orbitals=4,
        occupied_spin_orbitals=(0, 1),
        mapper_type=mapper,
    ) == expected


@pytest.mark.parametrize("mapper", ["jw", "bk", "parity"])
def test_factory_prepares_the_reference_state_in_the_selected_mapping(mapper):
    molecule = SimpleNamespace(
        mapper_type=mapper,
        n_qubits=4,
        n_electrons=2,
        sp_singles=[],
        sp_doubles=[],
        ccsd_full_parameter_count=0,
    )
    factory = UCCSDFactory(molecule)
    expected = encoded_occupied_qubits(
        n_spin_orbitals=4,
        occupied_spin_orbitals=(0, 1),
        mapper_type=mapper,
    )
    assert factory.mapper_type == mapper
    assert factory.occupied_qubits == expected


def _expected_terms(qubit_operator, n_qubits):
    result = {}
    for term, coefficient in qubit_operator.terms.items():
        chars = ["I"] * n_qubits
        for qubit, char in term:
            chars[int(qubit)] = str(char)
        result["".join(chars)] = float(np.imag(complex(coefficient)))
    return result


@pytest.mark.parametrize("mapper", ["bk", "parity"])
def test_factory_generator_uses_the_same_mapping_as_the_hamiltonian(mapper):
    molecule = SimpleNamespace(
        mapper_type=mapper,
        n_qubits=4,
        n_electrons=2,
        ccsd_full_parameter_count=2,
        ccsd_items=lambda indices: (
            [{"packed_index": 0, "amplitude": 0.0}],
            [],
        ),
    )
    factory = UCCSDFactory(molecule, selected_packed_indices=(0,))
    fermion_generator = canonical_singlet_operator(0, 4, 2)
    expected = _expected_terms(
        map_fermion_operator(
            fermion_generator,
            mapper_type=mapper,
            n_spin_orbitals=4,
        ),
        4,
    )
    actual = {term.pauli: term.coefficient for term in factory.generators[0].terms}
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("mapper", ["bk", "parity"])
def test_h2_mapped_hartree_fock_reference_matches_pyscf(mapper):
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermionpyscf")
    molecule = MolecularDataEngine(
        geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        basis="sto-3g",
        mapper_type=mapper,
    ).run(run_fci=False)
    factory = UCCSDFactory(molecule, construction_mode="jit")
    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
    reference_energy = estimator.evaluate(
        factory.build(np.zeros(factory.num_params)), molecule.hamiltonian_data
    )
    assert reference_energy == pytest.approx(molecule.hf_energy, abs=1e-8)
