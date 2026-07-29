"""Classical optimizers used by cqlib-VQE."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class OptimizerResult:
    """Result returned by a custom optimizer."""

    success: bool
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    message: str
    trajectory: list[float] = field(default_factory=list)


def _finite_float(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_interval(name: str, value: float, *, include_one: bool = False) -> float:
    result = _finite_float(name, value)
    upper_valid = result <= 1.0 if include_one else result < 1.0
    if result < 0.0 or not upper_valid:
        interval = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must be in {interval}")
    return result


def _parameter_vector(x0: np.ndarray) -> np.ndarray:
    result = np.asarray(x0, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("x0 must contain only finite values")
    return result


def _evaluate(func: ObjectiveFunction, x: np.ndarray) -> float:
    value = float(func(x))
    if not math.isfinite(value):
        raise FloatingPointError(f"objective function returned non-finite value {value}")
    return value


def _central_difference_gradient(
    func: ObjectiveFunction,
    x: np.ndarray,
    step: float,
) -> tuple[np.ndarray, int]:
    gradient = np.zeros_like(x)
    for index in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (_evaluate(func, plus) - _evaluate(func, minus)) / (2.0 * step)
    return gradient, 2 * x.size


class ClassicalOptimizer(ABC):
    """Base class for custom optimizers accepted by :class:`VQESolver`."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False):
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        self.max_iter = int(max_iter)
        self.tol = _nonnegative_float("tol", tol)
        self.verbose = bool(verbose)
        self.history: list[float] = []

    @abstractmethod
    def minimize(
        self,
        objective_function: ObjectiveFunction,
        x0: np.ndarray,
    ) -> OptimizerResult:
        """Minimize ``objective_function`` starting from ``x0``."""

    def _reset_history(self) -> None:
        self.history = []

    def _result(
        self,
        *,
        success: bool,
        x: np.ndarray,
        fun: float,
        nfev: int,
        nit: int,
        message: str,
    ) -> OptimizerResult:
        return OptimizerResult(
            success=bool(success),
            x=np.asarray(x, dtype=float).reshape(-1).copy(),
            fun=_finite_float("fun", fun),
            nfev=int(nfev),
            nit=int(nit),
            message=str(message),
            trajectory=self.history.copy(),
        )

    def _log_step(self, iteration: int, cost: float, params: np.ndarray) -> None:
        if self.verbose and iteration % 10 == 0:
            logger.info(
                "Iter %03d | Cost: %.8f | ParamNorm: %.4f",
                iteration,
                cost,
                np.linalg.norm(params),
            )


class MomentumOptimizer(ClassicalOptimizer):
    """Finite-difference gradient descent with heavy-ball momentum."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        max_iter: int = 200,
        tol: float = 1e-6,
        epsilon: float = 1e-5,
    ):
        super().__init__(max_iter, tol)
        self.lr = _positive_float("learning_rate", learning_rate)
        self.momentum = _unit_interval("momentum", momentum)
        self.epsilon = _positive_float("epsilon", epsilon)

    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizerResult:
        self._reset_history()
        x = _parameter_vector(x0)
        velocity = np.zeros_like(x)
        best_x = x.copy()
        best_fun = math.inf
        nfev = 0

        for iteration in range(self.max_iter):
            current_cost = _evaluate(func, x)
            nfev += 1
            self.history.append(current_cost)
            if current_cost < best_fun:
                best_fun = current_cost
                best_x = x.copy()

            if len(self.history) >= 2 and abs(self.history[-2] - self.history[-1]) <= self.tol:
                return self._result(
                    success=True,
                    x=best_x,
                    fun=best_fun,
                    nfev=nfev,
                    nit=iteration + 1,
                    message="Converged (tolerance)",
                )

            gradient, evaluations = _central_difference_gradient(func, x, self.epsilon)
            nfev += evaluations
            velocity = self.momentum * velocity + self.lr * gradient
            x = x - velocity
            self._log_step(iteration, current_cost, x)

        return self._result(
            success=False,
            x=best_x,
            fun=best_fun,
            nfev=nfev,
            nit=self.max_iter,
            message="Maximum iterations reached",
        )


class AdamOptimizer(ClassicalOptimizer):
    """Adam with central finite-difference gradients."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        super().__init__(max_iter, tol)
        self.lr = _positive_float("lr", lr)
        self.beta1 = _unit_interval("beta1", beta1)
        self.beta2 = _unit_interval("beta2", beta2)
        self.eps = _positive_float("epsilon", epsilon)
        self.fd_eps = 1e-5

    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizerResult:
        self._reset_history()
        x = _parameter_vector(x0)
        first_moment = np.zeros_like(x)
        second_moment = np.zeros_like(x)
        best_x = x.copy()
        best_fun = math.inf
        nfev = 0

        for iteration in range(self.max_iter):
            current_cost = _evaluate(func, x)
            nfev += 1
            self.history.append(current_cost)
            if current_cost < best_fun:
                best_fun = current_cost
                best_x = x.copy()

            if len(self.history) >= 11 and abs(self.history[-1] - self.history[-10]) <= self.tol:
                return self._result(
                    success=True,
                    x=best_x,
                    fun=best_fun,
                    nfev=nfev,
                    nit=iteration + 1,
                    message="Converged (stable cost)",
                )

            gradient, evaluations = _central_difference_gradient(func, x, self.fd_eps)
            nfev += evaluations
            step = iteration + 1
            first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * gradient
            second_moment = self.beta2 * second_moment + (1.0 - self.beta2) * gradient**2
            corrected_first = first_moment / (1.0 - self.beta1**step)
            corrected_second = second_moment / (1.0 - self.beta2**step)
            x = x - self.lr * corrected_first / (np.sqrt(corrected_second) + self.eps)
            self._log_step(iteration, current_cost, x)

        return self._result(
            success=False,
            x=best_x,
            fun=best_fun,
            nfev=nfev,
            nit=self.max_iter,
            message="Maximum iterations reached",
        )


class SPSAOptimizer(ClassicalOptimizer):
    """Simultaneous perturbation stochastic approximation."""

    def __init__(
        self,
        max_iter: int = 100,
        a: float = 0.6,
        c: float = 0.1,
        A: float = 0.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
    ):
        super().__init__(max_iter)
        self.a = _positive_float("a", a)
        self.c = _positive_float("c", c)
        self.A = _nonnegative_float("A", A)
        self.alpha = _positive_float("alpha", alpha)
        self.gamma = _positive_float("gamma", gamma)

    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizerResult:
        self._reset_history()
        theta = _parameter_vector(x0)
        best_theta = theta.copy()
        best_cost = _evaluate(func, theta)
        nfev = 1

        for iteration in range(self.max_iter):
            step = iteration + 1
            learning_rate = self.a / ((step + self.A) ** self.alpha)
            perturbation_size = self.c / (step**self.gamma)
            delta = 2 * np.random.randint(0, 2, size=theta.size) - 1

            theta_plus = theta + perturbation_size * delta
            theta_minus = theta - perturbation_size * delta
            cost_plus = _evaluate(func, theta_plus)
            cost_minus = _evaluate(func, theta_minus)
            nfev += 2

            gradient = (cost_plus - cost_minus) * delta / (2.0 * perturbation_size)
            theta = theta - learning_rate * gradient

            current_cost = _evaluate(func, theta)
            nfev += 1
            self.history.append(current_cost)
            if current_cost < best_cost:
                best_cost = current_cost
                best_theta = theta.copy()
            self._log_step(iteration, current_cost, theta)

        return self._result(
            success=True,
            x=best_theta,
            fun=best_cost,
            nfev=nfev,
            nit=self.max_iter,
            message="SPSA completed",
        )


class AdagradOptimizer(ClassicalOptimizer):
    """Adagrad with central finite-difference gradients."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        super().__init__(max_iter, tol)
        self.lr = _positive_float("learning_rate", learning_rate)
        self.eps = _positive_float("epsilon", epsilon)
        self.fd_eps = 1e-5

    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizerResult:
        self._reset_history()
        x = _parameter_vector(x0)
        accumulated_squares = np.zeros_like(x)
        best_x = x.copy()
        best_fun = math.inf
        nfev = 0

        for iteration in range(self.max_iter):
            current_cost = _evaluate(func, x)
            nfev += 1
            self.history.append(current_cost)
            if current_cost < best_fun:
                best_fun = current_cost
                best_x = x.copy()

            if len(self.history) >= 6 and abs(self.history[-1] - self.history[-5]) <= self.tol:
                return self._result(
                    success=True,
                    x=best_x,
                    fun=best_fun,
                    nfev=nfev,
                    nit=iteration + 1,
                    message="Converged (stable cost)",
                )

            gradient, evaluations = _central_difference_gradient(func, x, self.fd_eps)
            nfev += evaluations
            accumulated_squares += gradient**2
            x = x - self.lr * gradient / (np.sqrt(accumulated_squares) + self.eps)
            self._log_step(iteration, current_cost, x)

        return self._result(
            success=False,
            x=best_x,
            fun=best_fun,
            nfev=nfev,
            nit=self.max_iter,
            message="Maximum iterations reached",
        )


class NelderMeadOptimizer(ClassicalOptimizer):
    """Nelder-Mead simplex search."""

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ):
        super().__init__(max_iter, tol)
        self.alpha = _positive_float("alpha", alpha)
        self.gamma = _positive_float("gamma", gamma)
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")
        self.rho = _unit_interval("rho", rho)
        if self.rho == 0.0:
            raise ValueError("rho must be positive")
        self.sigma = _unit_interval("sigma", sigma)
        if self.sigma == 0.0:
            raise ValueError("sigma must be positive")

    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizerResult:
        self._reset_history()
        initial = _parameter_vector(x0)
        dimension = initial.size
        if dimension == 0:
            value = _evaluate(func, initial)
            self.history.append(value)
            return self._result(
                success=True,
                x=initial,
                fun=value,
                nfev=1,
                nit=0,
                message="No parameters",
            )

        simplex = [initial.copy()]
        for index in range(dimension):
            point = initial.copy()
            point[index] = point[index] + 0.05 if point[index] != 0.0 else 0.00025
            simplex.append(point)

        simplex_scores = [(_evaluate(func, point), point) for point in simplex]
        nfev = dimension + 1

        for iteration in range(self.max_iter):
            simplex_scores.sort(key=lambda item: item[0])
            best_score, best_point = simplex_scores[0]
            worst_score, worst_point = simplex_scores[-1]
            second_worst_score = simplex_scores[-2][0]
            self.history.append(best_score)

            score_spread = max(abs(score - best_score) for score, _ in simplex_scores)
            simplex_size = max(
                float(np.linalg.norm(point - best_point, ord=np.inf))
                for _, point in simplex_scores
            )
            coordinate_tolerance = math.sqrt(max(self.tol, np.finfo(float).eps))
            if score_spread <= self.tol and simplex_size <= coordinate_tolerance:
                return self._result(
                    success=True,
                    x=best_point,
                    fun=best_score,
                    nfev=nfev,
                    nit=iteration,
                    message="Converged (simplex standard deviation)",
                )

            centroid = np.mean([point for _, point in simplex_scores[:-1]], axis=0)
            reflected = centroid + self.alpha * (centroid - worst_point)
            reflected_score = _evaluate(func, reflected)
            nfev += 1

            if best_score <= reflected_score < second_worst_score:
                simplex_scores[-1] = (reflected_score, reflected)
            elif reflected_score < best_score:
                expanded = centroid + self.gamma * (reflected - centroid)
                expanded_score = _evaluate(func, expanded)
                nfev += 1
                simplex_scores[-1] = (
                    (expanded_score, expanded)
                    if expanded_score < reflected_score
                    else (reflected_score, reflected)
                )
            else:
                perform_shrink = False
                if reflected_score < worst_score:
                    contracted = centroid + self.rho * (reflected - centroid)
                    contracted_score = _evaluate(func, contracted)
                    nfev += 1
                    if contracted_score < reflected_score:
                        simplex_scores[-1] = (contracted_score, contracted)
                    else:
                        perform_shrink = True
                else:
                    contracted = centroid - self.rho * (reflected - centroid)
                    contracted_score = _evaluate(func, contracted)
                    nfev += 1
                    if contracted_score < worst_score:
                        simplex_scores[-1] = (contracted_score, contracted)
                    else:
                        perform_shrink = True

                if perform_shrink:
                    new_scores = [(best_score, best_point)]
                    for _, point in simplex_scores[1:]:
                        shrunk = best_point + self.sigma * (point - best_point)
                        new_scores.append((_evaluate(func, shrunk), shrunk))
                        nfev += 1
                    simplex_scores = new_scores

            self._log_step(iteration, best_score, best_point)

        simplex_scores.sort(key=lambda item: item[0])
        best_score, best_point = simplex_scores[0]
        return self._result(
            success=False,
            x=best_point,
            fun=best_score,
            nfev=nfev,
            nit=self.max_iter,
            message="Maximum iterations reached",
        )


class QuantumCoordinateDescent(ClassicalOptimizer):
    """Coordinate descent for objectives with one sinusoidal frequency per parameter.

    With all other parameters fixed, each coordinate must have the form
    ``A*cos(delta) + B*sin(delta) + C``. Both extrema separated by ``pi`` are
    evaluated, and the current point is retained if neither candidate improves it.
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6):
        super().__init__(max_iter, tol)

    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizerResult:
        self._reset_history()
        params = _parameter_vector(x0)
        best_params = params.copy()
        best_cost = _evaluate(func, params)
        nfev = 1

        if params.size == 0:
            self.history.append(best_cost)
            return self._result(
                success=True,
                x=best_params,
                fun=best_cost,
                nfev=nfev,
                nit=0,
                message="No parameters",
            )

        for iteration in range(self.max_iter):
            previous_cost = best_cost
            current_cost = best_cost

            for index in range(params.size):
                original = float(params[index])

                params[index] = original
                value_at_zero = _evaluate(func, params)
                params[index] = original + np.pi / 2.0
                value_at_plus = _evaluate(func, params)
                params[index] = original - np.pi / 2.0
                value_at_minus = _evaluate(func, params)
                nfev += 3

                offset = 0.5 * (value_at_plus + value_at_minus)
                cosine_coefficient = value_at_zero - offset
                sine_coefficient = 0.5 * (value_at_plus - value_at_minus)
                amplitude = math.hypot(cosine_coefficient, sine_coefficient)

                if amplitude <= 1e-12:
                    params[index] = original
                    current_cost = value_at_zero
                    continue

                phase = math.atan2(sine_coefficient, cosine_coefficient)
                candidate_one = original + phase
                candidate_two = candidate_one + np.pi

                params[index] = candidate_one
                value_one = _evaluate(func, params)
                params[index] = candidate_two
                value_two = _evaluate(func, params)
                nfev += 2

                current_cost, selected = min(
                    (value_at_zero, original),
                    (value_one, candidate_one),
                    (value_two, candidate_two),
                    key=lambda item: item[0],
                )
                params[index] = selected

            self.history.append(current_cost)
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()
            self._log_step(iteration, current_cost, params)

            if abs(previous_cost - current_cost) <= self.tol:
                return self._result(
                    success=True,
                    x=best_params,
                    fun=best_cost,
                    nfev=nfev,
                    nit=iteration + 1,
                    message="Converged (stable cost)",
                )

        return self._result(
            success=False,
            x=best_params,
            fun=best_cost,
            nfev=nfev,
            nit=self.max_iter,
            message="Maximum iterations reached",
        )
