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

import types

import numpy as np
import pytest

from cqlib_vqe.vqe.adaptive import AdaptiveSelectedUCCSDSolver, AdaptiveSelectionConfig


class Descriptor:
    def __init__(self, packed_index: int):
        self.packed_index = packed_index


def fake_molecule(amplitudes):
    return types.SimpleNamespace(
        ccsd_descriptors=[Descriptor(i) for i in range(len(amplitudes))],
        ccsd_parameters=np.asarray(amplitudes, dtype=float),
        ccsd_full_parameter_count=len(amplitudes),
        hamiltonian_data=[("I", 0.0)],
        fci_energy=None,
    )


def test_full_candidate_pool_keeps_exact_zero_ccsd_amplitudes():
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.12, 0.0, -0.03, 0.0]),
        estimator=object(),
        config=AdaptiveSelectionConfig(initial_amplitude_threshold=0.05),
    )

    assert solver.full_pool_indices == (0, 1, 2, 3)
    assert solver._pool() == [0, 1, 2, 3]
    assert solver._initial_selection(solver._pool()) == [0]
    assert solver._remaining_candidates([0]) == [1, 2, 3]


def test_nonzero_candidate_amplitude_threshold_is_rejected():
    with pytest.raises(ValueError, match="must be exactly 0.0"):
        AdaptiveSelectionConfig(candidate_amplitude_threshold=1e-6).validate()


def test_partial_candidate_screen_is_rejected_in_correctness_mode():
    with pytest.raises(ValueError, match="partial candidate scan"):
        AdaptiveSelectionConfig(max_screen_candidates=10).validate()


def test_zero_amplitude_generator_can_be_top_gradient_candidate(monkeypatch):
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.10, 0.0, 0.02]),
        estimator=object(),
        config=AdaptiveSelectionConfig(
            initial_amplitude_threshold=0.05,
            gradient_delta=1e-3,
        ),
    )

    # Avoid constructing real cqlib/OpenFermion factories.  The fake factory
    # only records which canonical packed indices are present.
    monkeypatch.setattr(
        solver,
        "_factory",
        lambda indices: types.SimpleNamespace(indices=tuple(sorted(indices))),
    )

    def fake_energy(factory, values):
        parameter_map = dict(zip(factory.indices, values, strict=True))
        # packed index 1 has classical amplitude 0.0 but a large variational
        # derivative dE/dtheta_1 = -0.1421095357.
        return -0.1421095357 * parameter_map.get(1, 0.0)

    monkeypatch.setattr(solver, "_energy", fake_energy)
    candidates = solver._screen(selected=[0], optimized=np.asarray([0.10]))

    assert [item.packed_index for item in candidates] == [1, 2]
    assert candidates[0].ccsd_amplitude == 0.0
    assert candidates[0].gradient == pytest.approx(-0.1421095357, abs=1e-12)
    assert candidates[0].absolute_gradient > candidates[1].absolute_gradient


def test_134_config_separates_pool_rounds_threshold_and_add_cap():
    config = AdaptiveSelectionConfig(
        pool_rounds=3,
        gradient_threshold=0.02,
        max_add_per_round=4,
    )
    config.validate()
    assert config.resolved_pool_rounds == 3
    assert config.resolved_max_add_per_round == 4


def test_134_compatibility_aliases_override_new_fields():
    config = AdaptiveSelectionConfig(
        pool_rounds=2,
        max_add_per_round=None,
        max_rounds=5,
        batch_size=1,
    )
    config.validate()
    assert config.resolved_pool_rounds == 5
    assert config.resolved_max_add_per_round == 1


def test_gradient_threshold_is_strict_and_default_adds_all_eligible():
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.1, 0.0, 0.0, 0.0]),
        estimator=object(),
        config=AdaptiveSelectionConfig(
            initial_amplitude_threshold=0.05,
            gradient_threshold=0.1,
            max_add_per_round=None,
        ),
    )
    candidates = [
        solver_gradient(1, 0.2),
        solver_gradient(2, 0.1),
        solver_gradient(3, 0.11),
    ]
    eligible = solver._eligible_candidates(candidates, active_count=1)
    assert [item.packed_index for item in eligible] == [1, 3]


def test_pool_rounds_count_scans_not_optimization_stages(monkeypatch):
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.1, 0.0, 0.0]),
        estimator=object(),
        config=AdaptiveSelectionConfig(
            initial_amplitude_threshold=0.05,
            pool_rounds=1,
            gradient_threshold=0.01,
            max_add_per_round=None,
            target_energy_error=0.0,
        ),
    )

    optimize_calls = []

    def fake_optimize(selected, initial, callback):
        optimize_calls.append(tuple(selected))
        return {
            "optimal_value": -float(len(selected)),
            "optimal_params": np.asarray(initial, dtype=float),
            "n_evals": 1,
        }

    monkeypatch.setattr(solver, "_optimize", fake_optimize)
    monkeypatch.setattr(
        solver,
        "_screen",
        lambda selected, optimized, **kwargs: [
            solver_gradient(1, 0.2),
            solver_gradient(2, 0.15),
        ],
    )

    result = solver.run(reference_energy=-10.0)
    assert optimize_calls == [(0,), (0, 1, 2)]
    assert result.pool_scans_completed == 1
    assert result.selected_packed_indices == (0, 1, 2)
    assert result.total_evaluations == 6  # 1 + 2*2 gradient probes + 1
    assert result.stop_reason == "complete canonical candidate pool exhausted"


def test_zero_pool_rounds_runs_only_initial_optimization(monkeypatch):
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.1, 0.0]),
        estimator=object(),
        config=AdaptiveSelectionConfig(
            initial_amplitude_threshold=0.05,
            pool_rounds=0,
            target_energy_error=0.0,
        ),
    )
    monkeypatch.setattr(
        solver,
        "_optimize",
        lambda selected, initial, callback: {
            "optimal_value": -1.0,
            "optimal_params": np.asarray(initial, dtype=float),
            "n_evals": 1,
        },
    )
    monkeypatch.setattr(
        solver,
        "_screen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not scan")),
    )
    result = solver.run(reference_energy=-10.0)
    assert result.pool_scans_completed == 0
    assert result.selected_packed_indices == (0,)
    assert result.stop_reason == "pool scan budget exhausted"


def solver_gradient(index: int, gradient: float):
    from cqlib_vqe.vqe.adaptive import CandidateGradient

    return CandidateGradient(
        packed_index=index,
        gradient=gradient,
        absolute_gradient=abs(gradient),
        ccsd_amplitude=0.0,
        energy_plus=0.0,
        energy_minus=0.0,
    )


def test_best_probe_initialization_reuses_existing_gradient_evaluations(monkeypatch):
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.1, 0.0]),
        estimator=object(),
        config=AdaptiveSelectionConfig(
            initial_amplitude_threshold=0.05,
            pool_rounds=1,
            gradient_delta=0.01,
            candidate_initialization="best_probe",
        ),
    )
    monkeypatch.setattr(
        solver,
        "_factory",
        lambda indices: types.SimpleNamespace(indices=tuple(sorted(indices))),
    )

    def energy(factory, values):
        mapping = dict(zip(factory.indices, values, strict=True))
        x = mapping.get(1, 0.0)
        return -1.0 + 0.5 * (x - 0.02) ** 2

    monkeypatch.setattr(solver, "_energy", energy)
    candidate = solver._screen(
        selected=[0],
        optimized=np.asarray([0.1]),
        base_energy=-1.0 + 0.5 * 0.02**2,
    )[0]
    assert candidate.initial_value == pytest.approx(0.01)
    assert candidate.predicted_improvement > 0.0


def test_native_pool_screen_compiles_one_complete_factory(monkeypatch):
    estimator = types.SimpleNamespace(evaluate_parameters=lambda *args, **kwargs: 0.0)
    solver = AdaptiveSelectedUCCSDSolver(
        fake_molecule([0.1, 0.0, 0.0, 0.0]),
        estimator=estimator,
        config=AdaptiveSelectionConfig(
            initial_amplitude_threshold=0.05,
            gradient_delta=1e-3,
            execution_mode="auto",
        ),
    )

    factory_calls = []

    def factory(indices):
        key = tuple(sorted(indices))
        factory_calls.append(key)
        return types.SimpleNamespace(
            indices=key,
            native_pauli_rotation_available=True,
            build_statevector=lambda values: values,
        )

    monkeypatch.setattr(solver, "_factory", factory)

    def energy(factory, values):
        # All probes must use the same complete 4-parameter factory.
        assert factory.indices == (0, 1, 2, 3)
        return float(np.dot(values, np.asarray([0.0, 1.0, 2.0, 3.0])))

    monkeypatch.setattr(solver, "_energy", energy)
    candidates = solver._screen(selected=[0], optimized=np.asarray([0.1]))

    assert factory_calls == [(0, 1, 2, 3)]
    assert [item.packed_index for item in candidates] == [3, 2, 1]
