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

"""VQE optimization loop with circuit and direct-statevector execution modes."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from utils.profiler import global_tracker

logger = logging.getLogger(__name__)


class VQESolver:
    """Coordinate a circuit factory, estimator, and classical optimizer.

    ``execution_mode="auto"`` honors ``construction_mode="bind"`` and uses
    cqlib2 ``assign_parameters``; otherwise it selects the fused direct-statevector
    path when available.  For COBYLA, the package uses a small chemistry-scale initial
    trust radius unless the caller supplies ``optimizer_options['rhobeg']``.
    SciPy's generic default of one radian is far too large for a CCSD-amplitude
    initial point, whose useful variations are normally much smaller.

    ``warm_start="gradient"`` is intended for exact local simulation only.  It
    computes a central finite-difference gradient at the classical initial
    point and performs a short backtracking line search before invoking the
    requested optimizer.  No gradient probe is hidden: all energies remain in
    ``history`` and are counted in ``n_evals``.
    """

    SUPPORTED_EXECUTION_MODES = {"auto", "circuit", "direct_statevector"}
    SUPPORTED_WARM_STARTS = {"none", "gradient"}

    def __init__(
        self,
        factory,
        estimator,
        optimizer_method: str | object = "COBYLA",
        max_iter: int = 50,
        *,
        tol: float = 1e-6,
        execution_mode: str = "auto",
        optimizer_options: Mapping[str, object] | None = None,
        warm_start: str = "none",
        warm_start_delta: float = 1e-3,
        warm_start_step: float = 0.05,
        warm_start_backtracks: int = 8,
    ) -> None:
        self.factory = factory
        self.estimator = estimator
        self.optimizer = optimizer_method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.execution_mode = self._validate_execution_mode(execution_mode)
        self.optimizer_options = dict(optimizer_options or {})
        self.warm_start = self._validate_warm_start(warm_start)
        self.warm_start_delta = float(warm_start_delta)
        self.warm_start_step = float(warm_start_step)
        self.warm_start_backtracks = int(warm_start_backtracks)
        if self.warm_start_delta <= 0:
            raise ValueError("warm_start_delta must be positive")
        if self.warm_start_step <= 0:
            raise ValueError("warm_start_step must be positive")
        if self.warm_start_backtracks <= 0:
            raise ValueError("warm_start_backtracks must be positive")
        self.history: list[float] = []
        self.best_history: list[float] = []
        self.parameter_history: list[np.ndarray] = []

    @classmethod
    def _validate_execution_mode(cls, mode: str) -> str:
        normalized = str(mode).lower()
        if normalized not in cls.SUPPORTED_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(cls.SUPPORTED_EXECUTION_MODES)}"
            )
        return normalized

    @classmethod
    def _validate_warm_start(cls, mode: str) -> str:
        normalized = str(mode).lower()
        if normalized not in cls.SUPPORTED_WARM_STARTS:
            raise ValueError(f"warm_start must be one of {sorted(cls.SUPPORTED_WARM_STARTS)}")
        return normalized

    def _resolved_execution_mode(self) -> str:
        if self.execution_mode == "auto":
            # Prefer the fused native Pauli statevector path when the installed
            # cqlib2 binding exposes it. The symbolic bind path remains available
            # explicitly through execution_mode="circuit".
            if (
                getattr(self.factory, "native_pauli_rotation_available", False)
                and hasattr(self.estimator, "evaluate_parameters")
                and hasattr(self.factory, "build_statevector")
            ):
                return "direct_statevector"
            if getattr(self.factory, "construction_mode", None) == "bind":
                return "circuit"
            if hasattr(self.estimator, "evaluate_parameters") and hasattr(
                self.factory, "build_statevector"
            ):
                return "direct_statevector"
            return "circuit"
        if self.execution_mode == "direct_statevector" and not (
            hasattr(self.estimator, "evaluate_parameters")
            and hasattr(self.factory, "build_statevector")
        ):
            raise TypeError(
                "direct_statevector mode requires estimator.evaluate_parameters(...) "
                "and factory.build_statevector(...)"
            )
        return self.execution_mode

    def _optimizer_options(self) -> dict[str, object]:
        options = {"maxiter": self.max_iter, **self.optimizer_options}
        if isinstance(self.optimizer, str) and self.optimizer.upper() == "COBYLA":
            options.setdefault("rhobeg", 0.03)
            options.setdefault("catol", 1e-8)
        return options

    def run(
        self,
        hamiltonian_data,
        callback: Callable[[np.ndarray, float, int], None] | None = None,
        *,
        initial_params: Sequence[float] | None = None,
    ) -> dict[str, object]:
        mode = self._resolved_execution_mode()
        if hasattr(self.estimator, "prepare_hamiltonian"):
            self.estimator.prepare_hamiltonian(
                hamiltonian_data, expected_num_qubits=getattr(self.factory, "n_qubits", None)
            )
        optimizer_name = (
            self.optimizer if isinstance(self.optimizer, str) else type(self.optimizer).__name__
        )
        logger.info("Starting VQE optimizer=%s execution=%s", optimizer_name, mode)

        self.history = []
        self.best_history = []
        self.parameter_history = []
        x0 = np.asarray(
            self.factory.initial_values if initial_params is None else initial_params,
            dtype=float,
        ).reshape(-1)
        if x0.size != int(self.factory.num_params):
            raise ValueError(
                f"Initial parameter mismatch: expected {self.factory.num_params}, got {x0.size}"
            )

        def objective(parameters) -> float:
            params = np.asarray(parameters, dtype=float).reshape(-1)
            with global_tracker.phase("Solver: Cost Function"):
                if mode == "direct_statevector":
                    energy = self.estimator.evaluate_parameters(
                        self.factory, params, hamiltonian_data
                    )
                else:
                    circuit = self.factory.build(params)
                    energy = self.estimator.evaluate(circuit, hamiltonian_data)
            value = float(np.real(energy))
            if not np.isfinite(value):
                raise FloatingPointError(f"VQE objective returned non-finite energy {value}")
            self.history.append(value)
            best = value if not self.best_history else min(self.best_history[-1], value)
            self.best_history.append(best)
            self.parameter_history.append(params.copy())
            if callback is not None:
                callback(params.copy(), value, len(self.history))
            return value

        start = perf_counter()

        if x0.size == 0:
            value = objective(x0)
            elapsed = perf_counter() - start
            return {
                "optimal_value": value,
                "optimal_params": x0.copy(),
                "n_evals": 1,
                "optimizer_n_evals": 0,
                "n_iters": 0,
                "time": elapsed,
                "success": True,
                "message": "No variational parameters; evaluated the reference state once.",
                "execution_mode": mode,
                "history": self.history.copy(),
                "best_history": self.best_history.copy(),
                "warm_start": {"mode": "none", "accepted": False},
            }

        warm_report: dict[str, object] = {"mode": self.warm_start, "accepted": False}
        start_point = x0.copy()
        if self.warm_start == "gradient":
            base_energy = objective(x0)
            gradient = np.zeros_like(x0)
            delta = self.warm_start_delta
            for index in range(x0.size):
                plus = x0.copy()
                minus = x0.copy()
                plus[index] += delta
                minus[index] -= delta
                gradient[index] = (objective(plus) - objective(minus)) / (2.0 * delta)
            gradient_norm = float(np.linalg.norm(gradient))
            warm_report.update(
                {
                    "base_energy": base_energy,
                    "gradient": gradient.copy(),
                    "gradient_norm": gradient_norm,
                }
            )
            if gradient_norm > np.finfo(float).eps:
                direction = -gradient / gradient_norm
                for backtrack in range(self.warm_start_backtracks):
                    step = self.warm_start_step * (0.5**backtrack)
                    candidate = x0 + step * direction
                    candidate_energy = objective(candidate)
                    if candidate_energy < base_energy:
                        start_point = candidate
                        warm_report.update(
                            {
                                "accepted": True,
                                "step": step,
                                "energy": candidate_energy,
                            }
                        )
                        break

        if isinstance(self.optimizer, str):
            result = minimize(
                fun=objective,
                x0=start_point,
                method=self.optimizer,
                tol=self.tol,
                options=self._optimizer_options(),
            )
            optimal_value = float(result.fun)
            optimal_params = np.asarray(result.x, dtype=float)
            optimizer_n_evals = int(getattr(result, "nfev", 0))
            n_iters = int(getattr(result, "nit", 0) or 0)
            success = bool(result.success)
            message = str(result.message)
        elif hasattr(self.optimizer, "minimize"):
            result = self.optimizer.minimize(objective, start_point)
            reported_value = float(result.fun)
            optimal_params = np.asarray(result.x, dtype=float)
            optimizer_n_evals = int(getattr(result, "nfev", 0))
            n_iters = int(getattr(result, "nit", 0) or 0)
            success = bool(getattr(result, "success", True))
            message = str(getattr(result, "message", "custom optimizer completed"))
            optimal_value = objective(optimal_params)
            if not np.isclose(
                reported_value,
                optimal_value,
                rtol=1e-9,
                atol=max(self.tol, 1e-12),
            ):
                logger.warning(
                    "Custom optimizer returned fun=%s, but objective(x)=%s; "
                    "using the verified value",
                    reported_value,
                    optimal_value,
                )
        else:
            raise TypeError("optimizer_method must be a SciPy method name or expose minimize()")

        elapsed = perf_counter() - start
        return {
            "optimal_value": optimal_value,
            "optimal_params": optimal_params,
            "n_evals": len(self.history),
            "optimizer_n_evals": optimizer_n_evals,
            "n_iters": n_iters,
            "time": elapsed,
            "success": success,
            "message": message,
            "execution_mode": mode,
            "history": self.history.copy(),
            "best_history": self.best_history.copy(),
            "warm_start": warm_report,
            "optimizer_options": self._optimizer_options(),
        }
