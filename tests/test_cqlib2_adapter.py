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

import numpy as np

from cqlib import Circuit
from cqlib.qis import Statevector

from cqlib_vqe.vqe.estimator import DirectStatevectorEstimator, NativeStatevectorEstimator
from cqlib_vqe.vqe.factory import UCCSDFactory
from cqlib_vqe.vqe.solver import VQESolver


def test_native_hamiltonian_converts_q0_left_to_cqlib_order():
    circuit = Circuit(2)
    circuit.x(0)
    estimator = NativeStatevectorEstimator(n_qubits=2)

    # Legacy "ZI" means Z on q0. cqlib2 needs native string "IZ".
    energy = estimator.evaluate(circuit, [("ZI", 1.0)])
    assert energy == -1.0


def test_jit_circuit_and_direct_statevector_are_equivalent():
    factory = UCCSDFactory.from_compiled(
        n_qubits=2,
        n_electrons=1,
        generators=[[("YI", 0.5)], [("XZ", -0.25)]],
        initial_values=[0.0, 0.0],
        construction_mode="jit",
    )
    params = np.array([0.37, -0.22])

    circuit_state = Statevector.from_circuit(factory.build(params))
    direct_state = factory.build_statevector(params)

    assert np.allclose(circuit_state.data, direct_state.data, atol=1e-12)


def test_symbolic_bind_path_matches_numeric_jit_path():
    generators = [[("YI", 0.5)], [("XZ", -0.25)]]
    params = np.array([0.11, 0.29])
    jit = UCCSDFactory.from_compiled(
        n_qubits=2,
        n_electrons=1,
        generators=generators,
        construction_mode="jit",
    )
    bound = UCCSDFactory.from_compiled(
        n_qubits=2,
        n_electrons=1,
        generators=generators,
        construction_mode="bind",
    )

    a = Statevector.from_circuit(jit.build(params)).data
    b = Statevector.from_circuit(bound.build(params)).data
    assert np.allclose(a, b, atol=1e-12)


def test_solver_auto_selects_direct_statevector_path():
    factory = UCCSDFactory.from_compiled(
        n_qubits=1,
        n_electrons=0,
        generators=[[("Y", 0.5)]],
        initial_values=[0.2],
    )
    estimator = DirectStatevectorEstimator(n_qubits=1)
    solver = VQESolver(
        factory,
        estimator,
        optimizer_method="COBYLA",
        max_iter=4,
        execution_mode="auto",
    )

    result = solver.run([("Z", 1.0)])
    assert result["execution_mode"] == "direct_statevector"
    assert np.isfinite(result["optimal_value"])
    assert result["n_evals"] >= 1


def test_pauli_rotation_sign_matches_exp_theta_generator():
    theta = 0.41
    coefficient = 0.5
    factory = UCCSDFactory.from_compiled(
        n_qubits=1,
        n_electrons=0,
        generators=[[("Y", coefficient)]],
        initial_values=[0.0],
    )
    state = factory.build_statevector([theta]).data

    # exp(theta * i*c*Y)|0> = cos(theta*c)|0> - sin(theta*c)|1>
    expected = np.array(
        [np.cos(theta * coefficient), -np.sin(theta * coefficient)], dtype=complex
    )
    assert np.allclose(state, expected, atol=1e-12)


def test_hamiltonian_cache_is_content_based_not_identity_based():
    circuit = Circuit(1)
    estimator = NativeStatevectorEstimator(n_qubits=1)
    terms = [("Z", 1.0)]
    assert estimator.evaluate(circuit, terms) == 1.0
    terms[0] = ("Z", -2.0)
    assert estimator.evaluate(circuit, terms) == -2.0


def test_cobyla_uses_chemistry_scale_default_rhobeg(monkeypatch):
    import types
    import vqe.solver as solver_module

    captured = {}

    def fake_minimize(*, fun, x0, method, tol, options):
        captured.update(method=method, tol=tol, options=dict(options))
        value = fun(np.asarray(x0, dtype=float))
        return types.SimpleNamespace(
            fun=value,
            x=np.asarray(x0, dtype=float),
            nfev=1,
            nit=0,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(solver_module, "minimize", fake_minimize)
    factory = UCCSDFactory.from_compiled(
        n_qubits=1,
        n_electrons=0,
        generators=[[('Y', 0.5)]],
        initial_values=[0.02],
    )
    estimator = DirectStatevectorEstimator(n_qubits=1)
    result = VQESolver(factory, estimator, max_iter=5).run([('Z', 1.0)])
    assert captured["options"]["rhobeg"] == 0.03
    assert captured["options"]["catol"] == 1e-8
    assert result["best_history"][-1] <= result["history"][0]


def test_gradient_warm_start_accepts_only_a_lower_energy_point(monkeypatch):
    import types
    import vqe.solver as solver_module

    class Factory:
        num_params = 1
        initial_values = np.array([1.0])

        def build(self, params):
            return np.asarray(params, dtype=float)

    class Estimator:
        def evaluate(self, circuit, _hamiltonian):
            x = float(circuit[0])
            return (x - 0.2) ** 2

    def fake_minimize(*, fun, x0, method, tol, options):
        value = fun(np.asarray(x0, dtype=float))
        return types.SimpleNamespace(
            fun=value,
            x=np.asarray(x0, dtype=float),
            nfev=1,
            nit=0,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(solver_module, "minimize", fake_minimize)
    result = VQESolver(
        Factory(),
        Estimator(),
        execution_mode="circuit",
        warm_start="gradient",
        warm_start_delta=1e-4,
        warm_start_step=0.1,
    ).run([('Z', 1.0)])
    assert result["warm_start"]["accepted"] is True
    assert result["warm_start"]["energy"] < result["warm_start"]["base_energy"]


def test_solver_auto_prefers_native_pauli_path_over_bind():
    factory = UCCSDFactory.from_compiled(
        n_qubits=1,
        n_electrons=0,
        generators=[[('Y', 0.5)]],
        initial_values=[0.2],
        construction_mode="bind",
    )
    estimator = DirectStatevectorEstimator(n_qubits=1)
    solver = VQESolver(
        factory,
        estimator,
        optimizer_method="COBYLA",
        max_iter=3,
        execution_mode="auto",
    )
    result = solver.run([('Z', 1.0)])
    assert factory.native_pauli_rotation_available is True
    assert result["execution_mode"] == "direct_statevector"
    assert np.isfinite(result["optimal_value"])


def test_explicit_circuit_mode_still_uses_bind_path():
    factory = UCCSDFactory.from_compiled(
        n_qubits=1,
        n_electrons=0,
        generators=[[('Y', 0.5)]],
        initial_values=[0.2],
        construction_mode="bind",
    )
    estimator = DirectStatevectorEstimator(n_qubits=1)
    result = VQESolver(
        factory,
        estimator,
        optimizer_method="COBYLA",
        max_iter=3,
        execution_mode="circuit",
    ).run([('Z', 1.0)])
    assert result["execution_mode"] == "circuit"
    assert np.isfinite(result["optimal_value"])
