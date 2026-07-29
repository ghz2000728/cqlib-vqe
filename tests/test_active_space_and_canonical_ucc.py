from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from cqlib_vqe.chemistry.active_space import ActiveSpaceConfig, select_active_space
from cqlib_vqe.chemistry.uccsd import (
    canonical_parameter_count,
    canonical_singlet_descriptors,
)
from cqlib_vqe.vqe.factory import UCCSDFactory


def test_canonical_closed_shell_parameter_counts_match_known_molecules():
    assert canonical_parameter_count(4, 2) == 2       # H2/STO-3G
    assert canonical_parameter_count(8, 4) == 14      # H4/STO-3G
    assert canonical_parameter_count(12, 4) == 44     # LiH/STO-3G
    assert canonical_parameter_count(14, 10) == 65    # H2O/STO-3G


def test_canonical_descriptor_order_is_unique_and_complete():
    descriptors = canonical_singlet_descriptors(12, 4)
    assert [item.packed_index for item in descriptors] == list(range(44))
    assert len({(item.kind, item.pair_a, item.pair_b) for item in descriptors}) == 44
    assert sum(item.kind == "single" for item in descriptors) == 8
    assert sum(item.kind == "double" for item in descriptors) == 36


def test_lih_automatic_frozen_core_produces_10_qubit_two_electron_problem():
    report = select_active_space(
        config=ActiveSpaceConfig.automatic(max_active_qubits=10, freeze_core=True),
        geometry=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))],
        n_orbitals=6,
        n_electrons=4,
        orbital_energies=[-2.4, -0.3, 0.1, 0.2, 0.3, 0.4],
    )
    assert report.frozen_occupied_indices == (0,)
    assert report.active_indices == (1, 2, 3, 4, 5)
    assert report.active_electrons == 2
    assert report.pre_taper_qubits == 10


def test_h2o_automatic_frozen_core_produces_12_qubit_eight_electron_problem():
    report = select_active_space(
        config=ActiveSpaceConfig.automatic(max_active_qubits=12, freeze_core=True),
        geometry=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.0)),
            ("H", (0.0, 1.0, 0.0)),
        ],
        n_orbitals=7,
        n_electrons=10,
        orbital_energies=[-20.0, -1.2, -0.8, -0.5, -0.3, 0.1, 0.2],
    )
    assert report.frozen_occupied_indices == (0,)
    assert report.active_indices == (1, 2, 3, 4, 5, 6)
    assert report.active_electrons == 8
    assert report.pre_taper_qubits == 12


def test_manual_space_cannot_silently_drop_an_occupied_orbital():
    with pytest.raises(ValueError, match="Every occupied orbital"):
        select_active_space(
            config=ActiveSpaceConfig.manual(),
            geometry=[("Be", (0.0, 0.0, 0.0))],
            n_orbitals=5,
            n_electrons=4,
            orbital_energies=[-4.0, -0.5, 0.1, 0.2, 0.3],
            occupied_indices=[0],
            active_indices=[2, 3, 4],
        )


def test_factory_requires_canonical_metadata_instead_of_old_merged_doubles():
    molecule = types.SimpleNamespace(
        mapper_type="jw",
        n_qubits=4,
        n_electrons=2,
        sp_singles=[{"indices": [0, 1], "amplitude": 0.1}],
        sp_doubles=[],
    )
    with pytest.raises(ValueError, match="canonical packed_index"):
        UCCSDFactory(molecule)


def test_factory_orders_parameters_by_packed_index(monkeypatch):
    class FakeOperator:
        def __init__(self, coefficient):
            self.terms = {((0, "Y"),): 1j * coefficient}

    fake_openfermion = types.ModuleType("openfermion")
    fake_openfermion.jordan_wigner = lambda operator: operator
    monkeypatch.setitem(sys.modules, "openfermion", fake_openfermion)

    import chemistry.uccsd as uccsd

    monkeypatch.setattr(
        uccsd,
        "canonical_singlet_operator",
        lambda packed_index, n_qubits, n_electrons: FakeOperator(0.5 + packed_index),
    )

    molecule = types.SimpleNamespace(
        mapper_type="jw",
        n_qubits=4,
        n_electrons=2,
        ccsd_full_parameter_count=2,
        sp_singles=[{"packed_index": 1, "amplitude": 0.2}],
        sp_doubles=[{"packed_index": 0, "amplitude": -0.3}],
    )
    factory = UCCSDFactory(molecule, trotter_order=1)
    assert factory.selected_packed_indices == (0, 1)
    assert np.allclose(factory.initial_values, [-0.3, 0.2])
    assert [item.label for item in factory.generators] == ["canonical_0", "canonical_1"]


def test_reduced_hamiltonian_receives_the_reported_frozen_and_active_indices():
    calls = []

    class FakeMolecule:
        def get_molecular_hamiltonian(self, *, occupied_indices, active_indices):
            calls.append((list(occupied_indices), list(active_indices)))
            return "interaction"

    from cqlib_vqe.chemistry.molecule import MolecularDataEngine

    engine = MolecularDataEngine.__new__(MolecularDataEngine)
    engine.molecule = FakeMolecule()
    engine.occupied_indices = [0]
    engine.active_indices = [1, 2, 3, 4, 5]
    assert engine._active_interaction_hamiltonian() == "interaction"
    assert calls == [([0], [1, 2, 3, 4, 5])]


def test_factory_can_restore_the_complete_canonical_pool(monkeypatch):
    class FakeOperator:
        def __init__(self, coefficient):
            self.terms = {((0, "Y"),): 1j * coefficient}

    fake_openfermion = types.ModuleType("openfermion")
    fake_openfermion.jordan_wigner = lambda operator: operator
    monkeypatch.setitem(sys.modules, "openfermion", fake_openfermion)

    import chemistry.uccsd as uccsd

    monkeypatch.setattr(
        uccsd,
        "canonical_singlet_operator",
        lambda packed_index, n_qubits, n_electrons: FakeOperator(0.5 + packed_index),
    )

    all_items = {
        0: {"packed_index": 0, "amplitude": -0.3},
        1: {"packed_index": 1, "amplitude": 2e-5},
    }

    def ccsd_items(indices):
        items = [all_items[index] for index in indices]
        return [item for item in items if item["packed_index"] == 0], [
            item for item in items if item["packed_index"] == 1
        ]

    molecule = types.SimpleNamespace(
        mapper_type="jw",
        n_qubits=4,
        n_electrons=2,
        ccsd_full_parameter_count=2,
        sp_singles=[all_items[0]],
        sp_doubles=[],
        ccsd_items=ccsd_items,
    )
    factory = UCCSDFactory(
        molecule,
        selected_packed_indices=[0, 1],
        trotter_order=1,
    )
    assert factory.selected_packed_indices == (0, 1)
    assert np.allclose(factory.initial_values, [-0.3, 2e-5])


def test_automatic_active_space_rejects_zero_minimum_virtual_orbitals():
    with pytest.raises(ValueError, match="min_virtual_orbitals must be at least 1"):
        ActiveSpaceConfig.automatic(min_virtual_orbitals=0).validate()
