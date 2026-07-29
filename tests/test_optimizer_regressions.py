from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy.optimize import minimize

from optim import (
    AdagradOptimizer,
    AdamOptimizer,
    MomentumOptimizer,
    NelderMeadOptimizer,
    OptimizerResult,
    QuantumCoordinateDescent,
    SPSAOptimizer,
)
from vqe import VQESolver


def test_spsa_result_value_matches_returned_parameters():
    objective = lambda x: float((x[0] - 1.0) ** 2)
    optimizer = SPSAOptimizer(max_iter=1)

    result = optimizer.minimize(objective, np.array([0.0]))

    assert result.x == pytest.approx([1.2])
    assert result.fun == pytest.approx(0.04)
    assert result.fun == pytest.approx(objective(result.x))
    assert result.nfev == 4


@pytest.mark.parametrize(
    ("objective", "x0", "expected"),
    [
        (lambda x: math.cos(x[0]), np.array([0.0]), -1.0),
        (lambda x: math.sin(x[0]), np.array([0.0]), -1.0),
        (lambda x: math.cos(x[0] - 0.4), np.array([0.2]), -1.0),
        (
            lambda x: math.cos(x[0]) + math.sin(x[1]),
            np.array([0.0, 0.0]),
            -2.0,
        ),
    ],
)
def test_quantum_coordinate_descent_finds_known_single_frequency_minima(
    objective,
    x0,
    expected,
):
    optimizer = QuantumCoordinateDescent(max_iter=4, tol=1e-10)

    result = optimizer.minimize(objective, x0)

    assert result.success
    assert result.fun == pytest.approx(expected, abs=1e-9)
    assert objective(result.x) == pytest.approx(result.fun, abs=1e-12)


def test_quantum_coordinate_descent_counts_every_objective_call():
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        return math.cos(x[0])

    result = QuantumCoordinateDescent(max_iter=2, tol=1e-12).minimize(
        objective,
        np.array([0.0]),
    )

    assert result.nfev == calls
    assert result.fun == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("dimension", "values", "expected_nfev"),
    [
        (2, [0.0, 5.0, 10.0, 3.0], 4),  # reflection
        (1, [1.0, 2.0, 0.0, -1.0], 4),  # expansion
        (1, [0.0, 10.0, 5.0, 4.0], 4),  # outside contraction
        (1, [0.0, 10.0, 11.0, 5.0], 4),  # inside contraction
        (1, [0.0, 10.0, 11.0, 12.0, 6.0], 5),  # shrink
    ],
)
def test_nelder_mead_branches_are_fully_initialized(dimension, values, expected_nfev):
    iterator = iter(values)

    def objective(_):
        return next(iterator)

    result = NelderMeadOptimizer(max_iter=1, tol=0.0).minimize(
        objective,
        np.zeros(dimension),
    )

    assert result.nfev == expected_nfev
    assert math.isfinite(result.fun)


def test_nelder_mead_agrees_with_scipy_on_a_standard_quadratic():
    objective = lambda x: float((x[0] - 1.5) ** 2 + 2.0 * (x[1] + 0.25) ** 2)
    x0 = np.array([4.0, -3.0])

    result = NelderMeadOptimizer(max_iter=500, tol=1e-10).minimize(objective, x0)
    reference = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-10, "fatol": 1e-10},
    )

    assert result.success
    assert result.x == pytest.approx(reference.x, abs=1e-5)
    assert result.fun == pytest.approx(reference.fun, abs=1e-10)


@pytest.mark.parametrize(
    ("factory", "objective", "x0"),
    [
        (
            lambda: MomentumOptimizer(max_iter=2, tol=0.0),
            lambda x: float((x[0] - 1.0) ** 2),
            np.array([0.0]),
        ),
        (
            lambda: AdamOptimizer(max_iter=2, tol=0.0),
            lambda x: float((x[0] - 1.0) ** 2),
            np.array([0.0]),
        ),
        (
            lambda: SPSAOptimizer(max_iter=2),
            lambda x: float((x[0] - 1.0) ** 2),
            np.array([0.0]),
        ),
        (
            lambda: AdagradOptimizer(max_iter=2, tol=0.0),
            lambda x: float((x[0] - 1.0) ** 2),
            np.array([0.0]),
        ),
        (
            lambda: NelderMeadOptimizer(max_iter=2, tol=0.0),
            lambda x: float((x[0] - 1.0) ** 2),
            np.array([0.0]),
        ),
        (
            lambda: QuantumCoordinateDescent(max_iter=2, tol=1e-12),
            lambda x: math.cos(x[0]),
            np.array([0.0]),
        ),
    ],
)
def test_optimizer_reuse_does_not_mutate_previous_result(factory, objective, x0):
    optimizer = factory()
    np.random.seed(7)
    first = optimizer.minimize(objective, x0)
    first_trajectory = first.trajectory.copy()

    np.random.seed(7)
    second = optimizer.minimize(objective, x0)

    assert first.trajectory == first_trajectory
    assert first.trajectory is not second.trajectory
    assert second.trajectory is not optimizer.history
    assert first.fun == pytest.approx(objective(first.x), abs=1e-10)
    assert second.fun == pytest.approx(objective(second.x), abs=1e-10)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: MomentumOptimizer(learning_rate=0.0),
        lambda: MomentumOptimizer(momentum=1.0),
        lambda: MomentumOptimizer(epsilon=0.0),
        lambda: AdamOptimizer(lr=0.0),
        lambda: AdamOptimizer(beta1=1.0),
        lambda: AdamOptimizer(beta2=-0.1),
        lambda: AdamOptimizer(epsilon=0.0),
        lambda: SPSAOptimizer(a=0.0),
        lambda: SPSAOptimizer(c=0.0),
        lambda: SPSAOptimizer(A=-1.0),
        lambda: AdagradOptimizer(learning_rate=0.0),
        lambda: AdagradOptimizer(epsilon=0.0),
        lambda: NelderMeadOptimizer(gamma=1.0),
        lambda: NelderMeadOptimizer(rho=0.0),
        lambda: NelderMeadOptimizer(sigma=1.0),
        lambda: QuantumCoordinateDescent(max_iter=0),
    ],
)
def test_optimizer_hyperparameters_are_validated(constructor):
    with pytest.raises(ValueError):
        constructor()


@dataclass
class _Factory:
    initial_values: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    num_params: int = 1
    construction_mode: str = "jit"
    n_qubits: int = 1

    def build(self, parameters):
        return np.asarray(parameters, dtype=float)


class _Estimator:
    def evaluate(self, parameters, _hamiltonian):
        return float((parameters[0] - 1.0) ** 2)


class _InconsistentOptimizer:
    def minimize(self, objective, x0):
        objective(x0)
        return OptimizerResult(
            success=True,
            x=np.array([1.0]),
            fun=123.0,
            nfev=1,
            nit=1,
            message="inconsistent test result",
        )


def test_vqe_solver_recomputes_custom_optimizer_result_value():
    solver = VQESolver(
        _Factory(),
        _Estimator(),
        optimizer_method=_InconsistentOptimizer(),
        max_iter=1,
    )

    result = solver.run([("Z", 1.0)])

    assert result["optimal_params"] == pytest.approx([1.0])
    assert result["optimal_value"] == pytest.approx(0.0)
    assert result["n_evals"] == 2
