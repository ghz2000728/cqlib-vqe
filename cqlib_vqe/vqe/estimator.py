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

"""Energy estimators for cqlib2.

The primary local path uses one native statevector simulation followed by one
native Hamiltonian expectation evaluation.  Sampling-based estimators are kept
for custom/cloud backends, but they are no longer used for exact local VQE.
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from cqlib import Circuit
from cqlib.ir import qcis
from cqlib.qis import Statevector

from ..utils.profiler import global_tracker
from .hamiltonian import (
    HamiltonianData,
    NativeHamiltonianCache,
    hamiltonian_cache_key,
    validate_hamiltonian_data,
)


class NativeStatevectorEstimator:
    """Exact local estimator using the cqlib2 QIS API.

    Per objective evaluation this performs exactly one
    ``Statevector.from_circuit`` call and one native Hamiltonian expectation.
    The converted cqlib2 Hamiltonian is cached by content.
    """

    def __init__(self, backend: Any = None, n_qubits: int | None = None) -> None:
        # ``backend`` is accepted only for backward compatibility with the old
        # FastStatevectorEstimator constructor.
        self.backend = backend
        self.n_qubits = n_qubits
        self._hamiltonian_cache = NativeHamiltonianCache()
        self._prepared_key = None
        self._prepared_hamiltonian = None

    def prepare_hamiltonian(
        self, hamiltonian_data: HamiltonianData, *, expected_num_qubits: int | None = None
    ):
        """Pre-convert a stable Hamiltonian for a high-frequency objective loop."""
        expected = self.n_qubits if expected_num_qubits is None else expected_num_qubits
        self._prepared_hamiltonian = self._hamiltonian_cache.get(
            hamiltonian_data, expected_num_qubits=expected
        )
        self._prepared_key = hamiltonian_cache_key(hamiltonian_data)
        return self._prepared_hamiltonian

    def evaluate(
        self,
        circuit_or_state: Circuit | Statevector,
        hamiltonian_data: HamiltonianData,
    ) -> float:
        with global_tracker.phase("Estimator: Native Statevector"):
            if isinstance(circuit_or_state, Statevector):
                state = circuit_or_state
            else:
                state = Statevector.from_circuit(circuit_or_state)

        if self.n_qubits is None:
            self.n_qubits = int(state.num_qubits)
        elif int(state.num_qubits) != self.n_qubits:
            raise ValueError(
                f"Statevector has {state.num_qubits} qubits, estimator expects {self.n_qubits}"
            )

        key = hamiltonian_cache_key(hamiltonian_data)
        if key == self._prepared_key and self._prepared_hamiltonian is not None:
            native_hamiltonian = self._prepared_hamiltonian
        else:
            native_hamiltonian = self._hamiltonian_cache.get(
                hamiltonian_data, expected_num_qubits=self.n_qubits
            )
            self._prepared_key = key
            self._prepared_hamiltonian = native_hamiltonian
        with global_tracker.phase("Estimator: Native Expectation"):
            return float(state.expectation(native_hamiltonian))

    def evaluate_statevector(self, state: Statevector, hamiltonian_data: HamiltonianData) -> float:
        return self.evaluate(state, hamiltonian_data)

    def clear_cache(self) -> None:
        self._hamiltonian_cache.clear()
        self._prepared_key = None
        self._prepared_hamiltonian = None


class DirectStatevectorEstimator(NativeStatevectorEstimator):
    """Fused parameter-to-energy path for factories with ``build_statevector``."""

    def evaluate_parameters(self, factory, parameters, hamiltonian_data: HamiltonianData) -> float:
        if not hasattr(factory, "build_statevector"):
            raise TypeError("factory does not implement build_statevector(parameters)")
        state = factory.build_statevector(parameters)
        return self.evaluate_statevector(state, hamiltonian_data)


class FastStatevectorEstimator(NativeStatevectorEstimator):
    """Backward-compatible alias for :class:`NativeStatevectorEstimator`."""


class ProbabilityBackend(Protocol):
    def run(self, circuit: Circuit) -> Mapping[str, float]: ...


class QWCMixin:
    @staticmethod
    def _validate_group_input(hamiltonian_data: HamiltonianData) -> None:
        validate_hamiltonian_data(hamiltonian_data)

    @staticmethod
    def _real_coefficient(coefficient) -> float:
        value = complex(coefficient)
        if abs(value.imag) > 1e-12:
            raise ValueError(
                "Sampling estimators require real coefficients for Hermitian Pauli terms"
            )
        return float(value.real)

    def group_hamiltonian(self, hamiltonian_data: HamiltonianData):
        """Greedy qubit-wise-commuting grouping for sampling backends."""
        self._validate_group_input(hamiltonian_data)
        groups: list[tuple[str, list[tuple[str, complex | float | int]]]] = []
        for pauli, coeff in sorted(hamiltonian_data, key=lambda item: abs(item[1]), reverse=True):
            for index, (master, terms) in enumerate(groups):
                if self._is_qwc_compatible(pauli, master):
                    groups[index] = (self._merge_pauli_strings(master, pauli), terms + [(pauli, coeff)])
                    break
            else:
                groups.append((pauli, [(pauli, coeff)]))
        return groups

    @staticmethod
    def _is_qwc_compatible(first: str, second: str) -> bool:
        return all(a == "I" or b == "I" or a == b for a, b in zip(first, second, strict=True))

    @staticmethod
    def _merge_pauli_strings(first: str, second: str) -> str:
        return "".join(a if a != "I" else b for a, b in zip(first, second, strict=True))

    @staticmethod
    def _apply_measurement_basis(circuit: Circuit, pauli: str) -> None:
        for qubit, char in enumerate(pauli):
            if char == "X":
                circuit.h(qubit)
            elif char == "Y":
                circuit.rx(qubit, np.pi / 2)

    @staticmethod
    def _compute_parity(probs: Mapping[str, float], pauli: str) -> float:
        """Compute parity assuming bitstrings use rightmost-bit = qubit 0."""
        mask = sum(1 << qubit for qubit, char in enumerate(pauli) if char != "I")
        expectation = 0.0
        for bitstring, probability in probs.items():
            if len(bitstring) != len(pauli) or any(bit not in "01" for bit in bitstring):
                raise ValueError(f"Invalid backend bitstring {bitstring!r} for {len(pauli)} qubits")
            parity = (int(bitstring, 2) & mask).bit_count() & 1
            expectation += float(probability) * (-1.0 if parity else 1.0)
        return expectation


class EnergyEstimator(QWCMixin):
    """Sampling estimator for a backend exposing ``run(circuit) -> probs``."""

    def __init__(self, backend: ProbabilityBackend) -> None:
        self.backend = backend

    def evaluate(self, base_circuit: Circuit, hamiltonian_data: HamiltonianData) -> float:
        total = 0.0
        with global_tracker.phase("Estimator: Sampled Full Eval"):
            for master, terms in self.group_hamiltonian(hamiltonian_data):
                measurement_circuit = deepcopy(base_circuit)
                if any(char not in "IZ" for char in master):
                    self._apply_measurement_basis(measurement_circuit, master)
                with global_tracker.phase("Backend: Simulation"):
                    probs = self.backend.run(measurement_circuit)
                for pauli, coeff in terms:
                    total += self._real_coefficient(coeff) * self._compute_parity(probs, pauli)
        return float(total)


class ParallelEnergyEstimator(EnergyEstimator):
    """Parallel QWC sampling for thread-safe C++ or remote backends."""

    def __init__(self, backend: ProbabilityBackend, max_workers: int = 8) -> None:
        super().__init__(backend)
        self.max_workers = int(max_workers)

    def evaluate(self, base_circuit: Circuit, hamiltonian_data: HamiltonianData) -> float:
        groups = self.group_hamiltonian(hamiltonian_data)

        def evaluate_group(group) -> float:
            master, terms = group
            measurement_circuit = deepcopy(base_circuit)
            if any(char not in "IZ" for char in master):
                self._apply_measurement_basis(measurement_circuit, master)
            probs = self.backend.run(measurement_circuit)
            return float(
                sum(
                    self._real_coefficient(coeff) * self._compute_parity(probs, pauli)
                    for pauli, coeff in terms
                )
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return float(sum(executor.map(evaluate_group, groups)))


class CloudBatchBackend(Protocol):
    def submit_experiment(self, *, circuit: list[str]) -> Any: ...
    def query_experiment(self, task_id: Any) -> Sequence[Mapping[str, Any]]: ...


class LegacyCloudEnergyEstimator(QWCMixin):
    """Adapter for the pre-2.0 submit_experiment/query_experiment API.

    Supply any backend implementing ``submit_experiment`` and
    ``query_experiment``.  Circuit serialization now uses
    ``cqlib.ir.qcis.dumps`` instead of the removed ``circuit.qcis`` property.
    """

    def __init__(self, backend: CloudBatchBackend, n_qubits: int, batch_size: int = 50) -> None:
        self.backend = backend
        self.n_qubits = int(n_qubits)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def evaluate(self, base_circuit: Circuit, hamiltonian_data: HamiltonianData) -> float:
        groups = self.group_hamiltonian(hamiltonian_data)
        base_qcis = qcis.dumps(base_circuit).rstrip()
        scripts: list[str] = []
        mappings: list[list[tuple[str, complex | float | int]]] = []

        with global_tracker.phase("Estimator: QCIS Assembly"):
            for master, terms in groups:
                suffix = [self._rotation_qcis(master), self._measurement_qcis()]
                scripts.append("\n".join(part for part in [base_qcis, *suffix] if part))
                mappings.append(terms)

        probabilities: list[Mapping[str, float]] = []
        with global_tracker.phase("Backend: Cloud Batch IO"):
            for start in range(0, len(scripts), self.batch_size):
                task_id = self.backend.submit_experiment(
                    circuit=scripts[start : start + self.batch_size]
                )
                results = self.backend.query_experiment(task_id)
                for result in results:
                    raw = result.get("probability", {})
                    probabilities.append(json.loads(raw) if isinstance(raw, str) else raw)

        if len(probabilities) != len(mappings):
            raise RuntimeError(
                f"Cloud backend returned {len(probabilities)} results for {len(mappings)} circuits"
            )

        total = 0.0
        for probs, terms in zip(probabilities, mappings, strict=True):
            for pauli, coeff in terms:
                total += self._real_coefficient(coeff) * self._compute_parity(probs, pauli)
        return float(total)

    @staticmethod
    def _rotation_qcis(pauli: str) -> str:
        commands = []
        for qubit, char in enumerate(pauli):
            if char == "X":
                commands.append(f"H Q{qubit}")
            elif char == "Y":
                commands.append(f"RX Q{qubit} {np.pi / 2:.17g}")
        return "\n".join(commands)

    def _measurement_qcis(self) -> str:
        return "M " + " ".join(f"Q{qubit}" for qubit in range(self.n_qubits))


class CloudEnergyEstimator(LegacyCloudEnergyEstimator):
    """Compatibility name for the pre-2.0 cloud API only.

    New ``cqlib_tianyan.TianyanBackend`` objects must be passed explicitly to
    :class:`cqlib_vqe.vqe.tianyan.TianyanEnergyEstimator`. Structural API detection was
    removed because choosing an execution protocol by guessing backend methods
    can hide integration errors.
    """

    def __init__(
        self, backend: CloudBatchBackend, n_qubits: int, batch_size: int = 50
    ) -> None:
        if callable(getattr(backend, "run_with_mode", None)):
            raise TypeError(
                "CloudEnergyEstimator is the legacy pre-2.0 adapter. "
                "Use TianyanEnergyEstimator explicitly for cqlib-tianyan backends."
            )
        super().__init__(backend, n_qubits, batch_size=batch_size)


class MutationEstimator(EnergyEstimator):
    """Compatibility fallback for the removed custom ``MutableCircuit`` path."""

    def __init__(self, backend: ProbabilityBackend) -> None:
        warnings.warn(
            "MutableCircuit is not part of cqlib2-dynamic. MutationEstimator now falls "
            "back to safe Circuit deepcopy; use NativeStatevectorEstimator locally.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(backend)


def __getattr__(name: str):
    if name in {
        "TianyanEnergyEstimator",
        "TianyanCloudEnergyEstimator",
        "TianyanMeasurementPlan",
        "TianyanEnergyResult",
    }:
        from . import tianyan as _tianyan

        return getattr(_tianyan, name)
    raise AttributeError(name)


__all__ = [
    "NativeStatevectorEstimator",
    "DirectStatevectorEstimator",
    "FastStatevectorEstimator",
    "EnergyEstimator",
    "ParallelEnergyEstimator",
    "LegacyCloudEnergyEstimator",
    "CloudEnergyEstimator",
    "MutationEstimator",
    "TianyanEnergyEstimator",
    "TianyanCloudEnergyEstimator",
    "TianyanMeasurementPlan",
    "TianyanEnergyResult",
]
