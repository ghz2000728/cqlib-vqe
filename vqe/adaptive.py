"""Threshold-grown canonical singlet-UCCSD with an unpruned candidate pool.

The classical CCSD amplitudes are used only to initialize the factorized
variational ansatz.  They never define the variational operator pool.  Each
operator-pool scan evaluates the complete canonical complement of the current
active ansatz, including generators whose classical CCSD amplitude is exactly
zero.

The public configuration uses full-pool semantics and supports a probe-based
initial value for newly selected parameters:

``initial_amplitude_threshold``
    Selects only the initial CCSD-seeded ansatz.
``pool_rounds``
    Maximum number of complete candidate-pool gradient scans.
``gradient_threshold``
    Every candidate with ``abs(gradient) > gradient_threshold`` is added after
    a scan, optionally capped by ``max_add_per_round``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .factory import UCCSDFactory
from .solver import VQESolver


@dataclass(frozen=True)
class AdaptiveSelectionConfig:
    """Configuration for threshold-grown selected-UCCSD.

    ``max_rounds`` and ``batch_size`` are retained as compatibility aliases.
    They map to ``pool_rounds`` and ``max_add_per_round`` respectively.
    New code should use the explicit field names.
    """

    initial_amplitude_threshold: float = 1e-3
    # Kept only so old callers fail loudly instead of silently restoring the
    # incorrect amplitude-pruned candidate-pool behavior.
    candidate_amplitude_threshold: float = 0.0
    min_initial_parameters: int = 1

    # Number of complete operator-pool scans after the initial optimization.
    # A value of zero performs only the original CCSD-seeded optimization.
    pool_rounds: int = 2
    # None means add every candidate whose gradient exceeds the threshold.
    max_add_per_round: int | None = None

    # Deprecated compatibility aliases.  When supplied, they override the new
    # fields so existing callers continue to run deterministically.
    max_rounds: int | None = None
    batch_size: int | None = None

    max_parameters: int | None = None
    # A partial candidate scan cannot certify gradient convergence.  The field
    # is retained for API compatibility but is rejected in correctness mode.
    max_screen_candidates: int | None = None
    gradient_delta: float = 1e-3
    gradient_threshold: float = 1e-4
    target_energy_error: float = 1.6e-3
    optimizer_method: str = "COBYLA"
    optimizer_maxiter: int = 80
    optimizer_tolerance: float = 1e-7
    optimizer_options: dict[str, Any] = field(
        default_factory=lambda: {"rhobeg": 0.03, "catol": 1e-8}
    )
    construction_mode: str = "bind"
    execution_mode: str = "auto"
    trotter_steps: int = 2
    trotter_order: int = 2
    # Selected parameters can start at the best already-evaluated probe
    # (+delta, -delta, or zero) without additional energy evaluations.
    candidate_initialization: str = "best_probe"

    @property
    def resolved_pool_rounds(self) -> int:
        return int(self.pool_rounds if self.max_rounds is None else self.max_rounds)

    @property
    def resolved_max_add_per_round(self) -> int | None:
        value = self.max_add_per_round if self.batch_size is None else self.batch_size
        return None if value is None else int(value)

    def validate(self) -> None:
        for name in (
            "initial_amplitude_threshold",
            "candidate_amplitude_threshold",
            "gradient_delta",
            "gradient_threshold",
            "target_energy_error",
            "optimizer_tolerance",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.candidate_amplitude_threshold != 0.0:
            raise ValueError(
                "candidate_amplitude_threshold must be exactly 0.0. Classical CCSD "
                "amplitudes may seed the initial ansatz, but they must not prune the "
                "adaptive canonical candidate pool."
            )
        if self.max_screen_candidates is not None:
            raise ValueError(
                "max_screen_candidates is disabled in correctness mode because a partial "
                "candidate scan cannot certify that all remaining canonical gradients are "
                "below threshold."
            )
        if self.gradient_delta <= 0:
            raise ValueError("gradient_delta must be positive")
        if self.min_initial_parameters <= 0:
            raise ValueError("min_initial_parameters must be positive")
        if self.resolved_pool_rounds < 0:
            raise ValueError("pool_rounds must be non-negative")
        max_add = self.resolved_max_add_per_round
        if max_add is not None and max_add <= 0:
            raise ValueError("max_add_per_round must be positive when supplied")
        if self.max_parameters is not None and self.max_parameters <= 0:
            raise ValueError("max_parameters must be positive when supplied")
        if self.optimizer_maxiter <= 0:
            raise ValueError("optimizer_maxiter must be positive")
        if self.trotter_steps <= 0:
            raise ValueError("trotter_steps must be positive")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        if self.construction_mode not in {"jit", "bind"}:
            raise ValueError("construction_mode must be 'jit' or 'bind'")
        if self.execution_mode not in {"auto", "circuit", "direct_statevector"}:
            raise ValueError(
                "execution_mode must be 'auto', 'circuit', or 'direct_statevector'"
            )
        if self.candidate_initialization not in {"zero", "best_probe"}:
            raise ValueError(
                "candidate_initialization must be 'zero' or 'best_probe'"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_pool_rounds"] = self.resolved_pool_rounds
        payload["resolved_max_add_per_round"] = self.resolved_max_add_per_round
        return payload


@dataclass(frozen=True)
class CandidateGradient:
    packed_index: int
    gradient: float
    absolute_gradient: float
    ccsd_amplitude: float
    energy_plus: float
    energy_minus: float
    initial_value: float = 0.0
    predicted_improvement: float = 0.0


@dataclass(frozen=True)
class AdaptiveRound:
    """One optimized ansatz state and, optionally, the following pool scan."""

    round_index: int
    selected_before: tuple[int, ...]
    candidate_pool_before: tuple[int, ...]
    optimized_energy: float
    reference_error: float | None
    optimizer_evaluations: int
    candidates: tuple[CandidateGradient, ...]
    gradient_norm: float | None
    added_indices: tuple[int, ...]
    wall_time_seconds: float
    pool_scan_index: int | None = None
    eligible_indices: tuple[int, ...] = ()
    selected_after: tuple[int, ...] = ()


@dataclass(frozen=True)
class AdaptiveSelectedUCCSDResult:
    optimal_value: float
    optimal_params: np.ndarray
    full_pool_indices: tuple[int, ...]
    initial_selected_packed_indices: tuple[int, ...]
    selected_packed_indices: tuple[int, ...]
    remaining_candidate_packed_indices: tuple[int, ...]
    rounds: tuple[AdaptiveRound, ...]
    stop_reason: str
    reference_energy: float | None
    reference_error: float | None
    precision_certified: bool
    total_evaluations: int
    full_parameter_count: int
    pool_scans_completed: int
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimal_value": float(self.optimal_value),
            "optimal_params": np.asarray(self.optimal_params, dtype=float).tolist(),
            "full_pool_indices": list(self.full_pool_indices),
            "initial_selected_packed_indices": list(self.initial_selected_packed_indices),
            "selected_packed_indices": list(self.selected_packed_indices),
            "remaining_candidate_packed_indices": list(
                self.remaining_candidate_packed_indices
            ),
            "stop_reason": self.stop_reason,
            "reference_energy": self.reference_energy,
            "reference_error": self.reference_error,
            "precision_certified": self.precision_certified,
            "total_evaluations": self.total_evaluations,
            "full_parameter_count": self.full_parameter_count,
            "pool_scans_completed": self.pool_scans_completed,
            "configuration": self.configuration,
            "rounds": [
                {
                    **asdict(item),
                    "candidates": [asdict(candidate) for candidate in item.candidates],
                }
                for item in self.rounds
            ],
        }


class AdaptiveSelectedUCCSDSolver:
    """Grow a canonical singlet-UCCSD ansatz using repeated full-pool scans."""

    def __init__(self, molecule_data, estimator, config: AdaptiveSelectionConfig | None = None):
        self.molecule = molecule_data
        self.estimator = estimator
        self.config = config or AdaptiveSelectionConfig()
        self.config.validate()

        descriptors = list(getattr(molecule_data, "ccsd_descriptors", ()))
        parameters = np.asarray(getattr(molecule_data, "ccsd_parameters", ()), dtype=float)
        if not descriptors or parameters.size != len(descriptors):
            raise ValueError("Molecule does not contain a complete canonical CCSD pool")

        descriptor_indices = tuple(int(item.packed_index) for item in descriptors)
        if len(set(descriptor_indices)) != len(descriptor_indices):
            raise ValueError("Canonical CCSD descriptors contain duplicate packed indices")
        expected_count = int(
            getattr(molecule_data, "ccsd_full_parameter_count", len(descriptors))
        )
        if len(descriptor_indices) != expected_count:
            raise ValueError(
                "Canonical descriptor list is not the complete pool: "
                f"{len(descriptor_indices)}/{expected_count}"
            )
        expected_indices = tuple(range(expected_count))
        if tuple(sorted(descriptor_indices)) != expected_indices:
            raise ValueError(
                "Canonical descriptor packed indices must cover the complete range "
                f"0..{expected_count - 1}"
            )

        self._amplitudes = {
            int(descriptor.packed_index): float(amplitude)
            for descriptor, amplitude in zip(descriptors, parameters, strict=True)
        }
        self.full_pool_indices: tuple[int, ...] = expected_indices
        self.initial_selected_indices: tuple[int, ...] = ()
        self.active_indices: tuple[int, ...] = ()
        self.candidate_indices: tuple[int, ...] = self.full_pool_indices
        self._factory_cache: dict[tuple[int, ...], UCCSDFactory] = {}
        if hasattr(self.estimator, "prepare_hamiltonian"):
            self.estimator.prepare_hamiltonian(
                self.molecule.hamiltonian_data,
                expected_num_qubits=getattr(self.molecule, "n_qubits", None),
            )

    def _pool(self) -> list[int]:
        """Backward-compatible accessor for the complete canonical pool."""
        return list(self.full_pool_indices)

    def _remaining_candidates(self, selected: Sequence[int]) -> list[int]:
        selected_set = {int(index) for index in selected}
        return [index for index in self.full_pool_indices if index not in selected_set]

    def _initial_selection(self, pool: Sequence[int]) -> list[int]:
        selected = [
            index
            for index in pool
            if abs(self._amplitudes[index]) >= self.config.initial_amplitude_threshold
        ]
        if len(selected) < self.config.min_initial_parameters:
            ranked = sorted(pool, key=lambda index: (-abs(self._amplitudes[index]), index))
            selected = sorted(
                set(selected).union(ranked[: self.config.min_initial_parameters])
            )
        if self.config.max_parameters is not None:
            selected = selected[: self.config.max_parameters]
        return sorted(selected)

    def _factory(self, indices: Sequence[int]) -> UCCSDFactory:
        key = tuple(sorted(int(index) for index in indices))
        if key not in self._factory_cache:
            self._factory_cache[key] = UCCSDFactory(
                self.molecule,
                construction_mode=self.config.construction_mode,
                trotter_steps=self.config.trotter_steps,
                trotter_order=self.config.trotter_order,
                selected_packed_indices=key,
            )
        return self._factory_cache[key]

    @staticmethod
    def _map(indices: Sequence[int], values: Sequence[float]) -> dict[int, float]:
        if len(indices) != len(values):
            raise ValueError("Packed indices and parameter vector have different lengths")
        return {
            int(index): float(value)
            for index, value in zip(indices, values, strict=True)
        }

    def _vector(
        self,
        indices: Sequence[int],
        current: dict[int, float],
        *,
        override: tuple[int, float] | None = None,
    ) -> np.ndarray:
        override_index = None if override is None else int(override[0])
        override_value = None if override is None else float(override[1])
        values = []
        for index in sorted(int(value) for value in indices):
            if index == override_index:
                values.append(override_value)
            elif index in current:
                values.append(current[index])
            else:
                # A newly appended generator starts at the identity.  This
                # preserves the optimized pre-growth state exactly.
                values.append(0.0)
        return np.asarray(values, dtype=float)

    def _resolved_execution_mode(self, factory: UCCSDFactory) -> str:
        mode = self.config.execution_mode
        if mode == "auto":
            if (
                getattr(factory, "native_pauli_rotation_available", False)
                and hasattr(self.estimator, "evaluate_parameters")
                and hasattr(factory, "build_statevector")
            ):
                return "direct_statevector"
            if getattr(factory, "construction_mode", None) == "bind":
                return "circuit"
            if hasattr(self.estimator, "evaluate_parameters") and hasattr(
                factory, "build_statevector"
            ):
                return "direct_statevector"
            return "circuit"
        return mode

    def _energy(self, factory: UCCSDFactory, values: np.ndarray) -> float:
        mode = self._resolved_execution_mode(factory)
        if mode == "direct_statevector":
            if not hasattr(self.estimator, "evaluate_parameters"):
                raise TypeError(
                    "direct_statevector mode requires estimator.evaluate_parameters(...)"
                )
            return float(
                self.estimator.evaluate_parameters(
                    factory, values, self.molecule.hamiltonian_data
                )
            )
        if mode == "circuit":
            # For a bind factory this calls cqlib2 Circuit.assign_parameters.
            circuit = factory.build(values)
            return float(self.estimator.evaluate(circuit, self.molecule.hamiltonian_data))
        raise ValueError(f"Unsupported execution mode {mode!r}")

    def _optimize(
        self,
        selected: Sequence[int],
        initial: np.ndarray,
        callback: Callable[[np.ndarray, float, int], None] | None,
    ) -> dict[str, Any]:
        factory = self._factory(selected)
        solver = VQESolver(
            factory,
            self.estimator,
            optimizer_method=self.config.optimizer_method,
            max_iter=self.config.optimizer_maxiter,
            tol=self.config.optimizer_tolerance,
            optimizer_options=self.config.optimizer_options,
            execution_mode=self.config.execution_mode,
        )
        return solver.run(
            self.molecule.hamiltonian_data,
            initial_params=initial,
            callback=callback,
        )

    def _screen(
        self,
        selected: Sequence[int],
        optimized: np.ndarray,
        pool: Sequence[int] | None = None,
        *,
        base_energy: float | None = None,
    ) -> list[CandidateGradient]:
        # `pool` is accepted for source compatibility, but correctness requires
        # screening the complete canonical complement of the active ansatz.
        if pool is not None and tuple(sorted(int(i) for i in pool)) != self.full_pool_indices:
            raise ValueError("Adaptive screening requires the complete canonical pool")
        remaining = self._remaining_candidates(selected)
        remaining.sort(key=lambda index: (-abs(self._amplitudes[index]), index))
        current = self._map(selected, optimized)
        delta = self.config.gradient_delta
        candidates: list[CandidateGradient] = []

        # For local statevector execution, compile the complete canonical pool
        # exactly once. Inactive generators receive zero parameters and are
        # skipped by UCCSDFactory before entering the Rust kernel. This removes
        # one OpenFermion/factory compilation per candidate. Circuit/hardware
        # execution retains the selected+candidate construction.
        full_factory = self._factory(self.full_pool_indices)
        use_full_factory = self._resolved_execution_mode(full_factory) == "direct_statevector"
        if use_full_factory:
            base_vector = self._vector(self.full_pool_indices, current)

        for candidate in remaining:
            if use_full_factory:
                plus = base_vector.copy()
                minus = base_vector.copy()
                plus[candidate] = +delta
                minus[candidate] = -delta
                factory = full_factory
            else:
                trial = sorted([*selected, candidate])
                factory = self._factory(trial)
                plus = self._vector(trial, current, override=(candidate, +delta))
                minus = self._vector(trial, current, override=(candidate, -delta))

            e_plus = self._energy(factory, plus)
            e_minus = self._energy(factory, minus)
            gradient = (e_plus - e_minus) / (2.0 * delta)

            initial_value = 0.0
            predicted_improvement = 0.0
            if (
                self.config.candidate_initialization == "best_probe"
                and base_energy is not None
            ):
                best_energy, initial_value = min(
                    (
                        (float(base_energy), 0.0),
                        (float(e_plus), +delta),
                        (float(e_minus), -delta),
                    ),
                    key=lambda item: item[0],
                )
                predicted_improvement = max(0.0, float(base_energy) - best_energy)

            candidates.append(
                CandidateGradient(
                    packed_index=int(candidate),
                    gradient=float(gradient),
                    absolute_gradient=float(abs(gradient)),
                    ccsd_amplitude=float(self._amplitudes[candidate]),
                    energy_plus=float(e_plus),
                    energy_minus=float(e_minus),
                    initial_value=float(initial_value),
                    predicted_improvement=float(predicted_improvement),
                )
            )
        candidates.sort(key=lambda item: (-item.absolute_gradient, item.packed_index))
        return candidates

    def _eligible_candidates(
        self, candidates: Sequence[CandidateGradient], active_count: int
    ) -> list[CandidateGradient]:
        # The user-facing rule is strictly greater than the threshold.
        eligible = [
            candidate
            for candidate in candidates
            if candidate.absolute_gradient > self.config.gradient_threshold
        ]
        max_add = self.config.resolved_max_add_per_round
        if max_add is not None:
            eligible = eligible[:max_add]
        if self.config.max_parameters is not None:
            slots = max(0, self.config.max_parameters - active_count)
            eligible = eligible[:slots]
        return eligible

    def run(
        self,
        *,
        reference_energy: float | None = None,
        round_callback: Callable[[AdaptiveRound], None] | None = None,
        optimizer_callback: Callable[[np.ndarray, float, int], None] | None = None,
    ) -> AdaptiveSelectedUCCSDResult:
        full_pool = list(self.full_pool_indices)
        selected = self._initial_selection(full_pool)
        self.initial_selected_indices = tuple(selected)
        self.active_indices = tuple(selected)
        self.candidate_indices = tuple(self._remaining_candidates(selected))
        current = {index: self._amplitudes[index] for index in selected}
        reference = (
            float(reference_energy)
            if reference_energy is not None
            else (
                None
                if getattr(self.molecule, "fci_energy", None) is None
                else float(self.molecule.fci_energy)
            )
        )

        rounds: list[AdaptiveRound] = []
        total_evaluations = 0
        pool_scans_completed = 0
        final_energy = float("inf")
        final_params = np.asarray([], dtype=float)
        stop_reason = "pool scan budget exhausted"
        pool_rounds = self.config.resolved_pool_rounds

        # One initial optimization plus one re-optimization after every allowed
        # pool scan.  Therefore pool_rounds=N permits at most N scans and N+1
        # optimization stages.
        for optimization_round in range(1, pool_rounds + 2):
            started = perf_counter()
            selected = sorted(selected)
            optimized_selection = tuple(selected)
            candidate_pool_before = tuple(self._remaining_candidates(selected))
            self.active_indices = optimized_selection
            self.candidate_indices = candidate_pool_before

            initial = self._vector(selected, current)
            optimization = self._optimize(selected, initial, optimizer_callback)
            final_energy = float(optimization["optimal_value"])
            final_params = np.asarray(optimization["optimal_params"], dtype=float)
            total_evaluations += int(optimization["n_evals"])
            current = self._map(selected, final_params)
            reference_error = None if reference is None else abs(final_energy - reference)

            candidates: list[CandidateGradient] = []
            eligible: list[CandidateGradient] = []
            added: list[int] = []
            gradient_norm: float | None = None
            pool_scan_index: int | None = None

            if reference_error is not None and reference_error <= self.config.target_energy_error:
                stop_reason = "reference-energy precision target reached"
            elif not candidate_pool_before:
                stop_reason = "complete canonical candidate pool exhausted"
            elif self.config.max_parameters is not None and len(selected) >= self.config.max_parameters:
                stop_reason = "maximum parameter budget reached"
            elif pool_scans_completed >= pool_rounds:
                stop_reason = "pool scan budget exhausted"
            else:
                pool_scan_index = pool_scans_completed + 1
                candidates = self._screen(
                    selected, final_params, base_energy=final_energy
                )
                pool_scans_completed += 1
                total_evaluations += 2 * len(candidates)
                gradient_norm = float(
                    np.linalg.norm([candidate.gradient for candidate in candidates])
                )
                eligible = self._eligible_candidates(candidates, len(selected))
                if not eligible:
                    stop_reason = "all remaining canonical gradients at or below threshold"
                else:
                    added = [candidate.packed_index for candidate in eligible]
                    selected = sorted(set(selected).union(added))
                    initial_by_index = {
                        candidate.packed_index: candidate.initial_value
                        for candidate in eligible
                    }
                    for index in added:
                        current[index] = float(initial_by_index[index])
                    self.active_indices = tuple(selected)
                    self.candidate_indices = tuple(self._remaining_candidates(selected))
                    stop_reason = "pool scan added candidates"

            item = AdaptiveRound(
                round_index=optimization_round,
                selected_before=optimized_selection,
                candidate_pool_before=candidate_pool_before,
                optimized_energy=final_energy,
                reference_error=reference_error,
                optimizer_evaluations=int(optimization["n_evals"]),
                candidates=tuple(candidates),
                gradient_norm=gradient_norm,
                added_indices=tuple(added),
                wall_time_seconds=perf_counter() - started,
                pool_scan_index=pool_scan_index,
                eligible_indices=tuple(candidate.packed_index for candidate in eligible),
                selected_after=tuple(selected),
            )
            rounds.append(item)
            if round_callback is not None:
                round_callback(item)
            if not added:
                break

        final_indices = tuple(sorted(selected))
        remaining_indices = tuple(self._remaining_candidates(final_indices))
        self.active_indices = final_indices
        self.candidate_indices = remaining_indices
        reference_error = None if reference is None else abs(final_energy - reference)
        return AdaptiveSelectedUCCSDResult(
            optimal_value=final_energy,
            optimal_params=final_params,
            full_pool_indices=self.full_pool_indices,
            initial_selected_packed_indices=self.initial_selected_indices,
            selected_packed_indices=final_indices,
            remaining_candidate_packed_indices=remaining_indices,
            rounds=tuple(rounds),
            stop_reason=stop_reason,
            reference_energy=reference,
            reference_error=reference_error,
            precision_certified=(
                reference_error is not None
                and reference_error <= self.config.target_energy_error
            ),
            total_evaluations=total_evaluations,
            full_parameter_count=len(self.full_pool_indices),
            pool_scans_completed=pool_scans_completed,
            configuration=self.config.to_dict(),
        )


__all__ = [
    "AdaptiveSelectionConfig",
    "CandidateGradient",
    "AdaptiveRound",
    "AdaptiveSelectedUCCSDResult",
    "AdaptiveSelectedUCCSDSolver",
]
