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

"""Tianyan quantum-hardware estimator for cqlib 2.x.

This module integrates the standalone :mod:`cqlib_tianyan` package with the
cqlib-vqe estimator interface.  Hamiltonian terms are greedily grouped by
qubit-wise commutativity (QWC), one measurement circuit is generated per group,
and all group circuits are submitted in a single Tianyan batch call.  The
cqlib-tianyan client transparently splits batches larger than the platform
limit and returns a single ``TaskHandle``.

The module intentionally uses structural typing and imports ``cqlib_tianyan``
only inside ``from_credentials``.  Local statevector users therefore do not
need to install the hardware client.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Protocol, Sequence

from cqlib import Circuit
from cqlib.compile import compile as compile_circuit
from cqlib.ir import qcis

from utils.profiler import global_tracker
from .estimator import QWCMixin
from .hamiltonian import HamiltonianData, validate_hamiltonian_data

_QUBIT_TOKEN = re.compile(r"\bQ(\d+)\b")
_MEASUREMENT_LINE = re.compile(r"(?m)^\s*M(?:\s|$)")
_VALID_CALIBRATION_MODES = {"enabled", "disabled"}
_VALID_GROUPING_MODES = {"qwc", "none"}
_TIANYAN_NATIVE_BASIS = ("RZ", "X2P", "X2M", "Y2P", "Y2M", "CZ")
_TIANYAN_NATIVE_QCIS_GATES = {
    "RZ", "X2P", "X2M", "Y2P", "Y2M", "XY2P", "XY2M", "CZ", "M"
}


class TianyanTask(Protocol):
    task_ids: Sequence[str]

    def wait(
        self, timeout_secs: float, poll_interval_secs: float = 5.0
    ) -> Sequence[Any]: ...


class TianyanBackendLike(Protocol):
    name: str
    num_qubits: int | None

    def run_with_mode(
        self, circuits: list[str], shots: int, mode: str = "auto"
    ) -> TianyanTask: ...


@dataclass(frozen=True)
class TianyanMeasurementGroup:
    """One QWC group and its corresponding hardware measurement metadata."""

    master_pauli: str
    terms: tuple[tuple[str, complex | float | int], ...]
    measured_logical_qubits: tuple[int, ...]
    measured_physical_qubits: tuple[int, ...]


@dataclass(frozen=True)
class TianyanMeasurementPlan:
    """Fully assembled hardware circuits for one VQE energy evaluation."""

    circuits: tuple[str, ...]
    groups: tuple[TianyanMeasurementGroup, ...]
    constant_energy: float
    logical_to_physical: tuple[int, ...]
    n_qubits: int

    @property
    def num_measurement_circuits(self) -> int:
        return len(self.circuits)

    @property
    def num_pauli_terms(self) -> int:
        return sum(len(group.terms) for group in self.groups)


@dataclass(frozen=True)
class TianyanTermResult:
    pauli: str
    coefficient: float
    expectation: float
    contribution: float


@dataclass(frozen=True)
class TianyanGroupResult:
    master_pauli: str
    task_id: str
    measured_physical_qubits: tuple[int, ...]
    terms: tuple[TianyanTermResult, ...]
    contribution: float


@dataclass(frozen=True)
class TianyanEnergyResult:
    """Detailed result returned by :meth:`TianyanEnergyEstimator.evaluate_with_details`."""

    energy: float
    constant_energy: float
    groups: tuple[TianyanGroupResult, ...]
    task_ids: tuple[str, ...]
    shots: int
    device_name: str
    calibration_mode: str


@dataclass(frozen=True)
class TianyanSubmittedEvaluation:
    """A submitted measurement plan and its live Tianyan task handle."""

    plan: TianyanMeasurementPlan
    task: TianyanTask | None


class TianyanEnergyEstimator(QWCMixin):
    """QWC-grouped VQE energy estimator for ``cqlib_tianyan`` backends.

    Parameters
    ----------
    backend:
        A ``cqlib_tianyan.TianyanBackend`` instance.
    n_qubits:
        Number of logical qubits in the VQE circuit and Hamiltonian.
    shots:
        Shots per QWC measurement circuit.
    calibration_mode:
        ``"enabled"`` or ``"disabled"``. Strict execution rejects ``"auto"``
        because the hardware SDK may silently fall back to raw counts.
    physical_qubits:
        Optional logical-to-physical mapping.  Entry ``i`` is the Tianyan
        physical qubit used for logical qubit ``i``.  When omitted, the identity
        mapping ``0..n_qubits-1`` is used.
    measure_only_support:
        Measure only qubits touched by the group's master Pauli string.  This is
        both faster and less noisy than measuring the full register and is the
        recommended default.
    grouping:
        ``"qwc"`` for grouped measurement or ``"none"`` for one circuit per
        Pauli term. Use ``"none"`` as the truth-revealing hardware reference.
    timeout_secs, poll_interval_secs:
        Polling settings passed to ``TaskHandle.wait``.

    Notes
    -----
    This estimator performs measurement grouping and batching, but it does not
    invent a hardware routing strategy.  The supplied ``physical_qubits`` must
    be compatible with the two-qubit interactions in the serialized circuit,
    or the circuit must already have been compiled/routed upstream.
    """

    def __init__(
        self,
        backend: TianyanBackendLike,
        n_qubits: int,
        *,
        shots: int = 4096,
        calibration_mode: str = "disabled",
        physical_qubits: Sequence[int] | None = None,
        measure_only_support: bool = True,
        grouping: str = "qwc",
        timeout_secs: float = 1800.0,
        poll_interval_secs: float = 5.0,
    ) -> None:
        self.backend = backend
        self.n_qubits = int(n_qubits)
        self.shots = int(shots)
        self.calibration_mode = str(calibration_mode).lower()
        self.measure_only_support = bool(measure_only_support)
        self.grouping = str(grouping).lower()
        self.timeout_secs = float(timeout_secs)
        self.poll_interval_secs = float(poll_interval_secs)

        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if self.calibration_mode not in _VALID_CALIBRATION_MODES:
            raise ValueError(
                "strict Tianyan execution accepts only calibration_mode='enabled' "
                "or 'disabled'. The SDK's 'auto' mode can silently fall back "
                "to raw counts, so it is intentionally rejected."
            )
        if self.grouping not in _VALID_GROUPING_MODES:
            raise ValueError(
                f"grouping must be one of {sorted(_VALID_GROUPING_MODES)}"
            )
        if self.timeout_secs <= 0:
            raise ValueError("timeout_secs must be positive")
        if self.poll_interval_secs <= 0:
            raise ValueError("poll_interval_secs must be positive")

        if physical_qubits is None:
            mapping = tuple(range(self.n_qubits))
        else:
            mapping = tuple(int(qubit) for qubit in physical_qubits)
        if len(mapping) != self.n_qubits:
            raise ValueError(
                f"physical_qubits must contain {self.n_qubits} entries, got {len(mapping)}"
            )
        if any(qubit < 0 for qubit in mapping):
            raise ValueError("physical qubit indices must be non-negative")
        if len(set(mapping)) != len(mapping):
            raise ValueError("physical_qubits must be one-to-one")
        self.logical_to_physical = mapping

        self.last_task: TianyanTask | None = None
        self.last_plan: TianyanMeasurementPlan | None = None
        self.last_result: TianyanEnergyResult | None = None
        self._platform: Any = None
        self._grouping_cache: tuple[
            tuple[tuple[str, complex], ...],
            float,
            tuple[tuple[str, tuple[tuple[str, complex], ...]], ...],
        ] | None = None

    @classmethod
    def from_credentials(
        cls,
        *,
        device_name: str,
        n_qubits: int,
        api_key: str | None = None,
        domain: str | None = None,
        save_credentials: bool | None = None,
        auto_refresh: bool | None = None,
        credentials_path: str | None = None,
        **estimator_kwargs: Any,
    ) -> "TianyanEnergyEstimator":
        """Create an estimator from a new API-key login or saved credentials.

        Passing ``api_key`` calls ``TianyanPlatform.login``.  Omitting it calls
        ``TianyanPlatform.from_credentials``.
        """

        try:
            from cqlib_tianyan import TianyanPlatform
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "Tianyan hardware support requires the optional package "
                "'cqlib-tianyan'. Install it with "
                "`pip install \"cqlib-vqe[tianyan]\"`."
            ) from exc

        platform_options = {
            key: value
            for key, value in {
                "domain": domain,
                "save_credentials": save_credentials,
                "auto_refresh": auto_refresh,
                "credentials_path": credentials_path,
            }.items()
            if value is not None
        }
        if api_key is None:
            platform = TianyanPlatform.from_credentials(**platform_options)
        else:
            normalized_api_key = str(api_key).strip()
            if not normalized_api_key:
                raise ValueError("api_key must not be empty or whitespace-only")
            platform = TianyanPlatform.login(normalized_api_key, **platform_options)
        backend = platform.get_backend(device_name)
        estimator = cls(backend, n_qubits, **estimator_kwargs)
        # Retain the authenticated platform for callers that want discovery or
        # device configuration without performing another login.
        estimator._platform = platform
        return estimator

    @property
    def device_name(self) -> str:
        return str(getattr(self.backend, "name", "unknown"))

    def validate_backend_mapping(self, *, require_available: bool = True) -> Any:
        """Fetch device configuration and validate the selected physical qubits.

        This method performs network I/O through ``backend.device_config()`` and
        is intentionally opt-in.  It checks backend availability when the API
        exposes ``is_available`` and verifies that every physical qubit in the
        logical mapping is present in the returned device topology.
        """

        is_available = getattr(self.backend, "is_available", None)
        if require_available and callable(is_available) and not is_available():
            raise RuntimeError(f"Tianyan backend {self.device_name!r} is not available")
        device_config = getattr(self.backend, "device_config", None)
        if not callable(device_config):
            raise TypeError("backend does not expose device_config()")
        device = device_config()
        topology = getattr(device, "topology", None)
        contains_qubit = getattr(topology, "contains_qubit", None)
        if callable(contains_qubit):
            missing = [
                qubit
                for qubit in self.logical_to_physical
                if not contains_qubit(qubit)
            ]
        else:
            available = {
                self._qubit_index(qubit)
                for qubit in getattr(device, "qubits", ())
            }
            missing = [
                qubit for qubit in self.logical_to_physical if qubit not in available
            ]
        if missing:
            raise ValueError(
                f"physical qubits are not present on backend {self.device_name}: {missing}"
            )
        return device

    def prepare(
        self, base_circuit: Circuit, hamiltonian_data: HamiltonianData
    ) -> TianyanMeasurementPlan:
        """Group the Hamiltonian and assemble all QCIS measurement circuits."""

        validate_hamiltonian_data(hamiltonian_data)
        circuit_qubits = int(getattr(base_circuit, "num_qubits"))
        if circuit_qubits != self.n_qubits:
            raise ValueError(
                f"Circuit has {circuit_qubits} qubits, estimator expects {self.n_qubits}"
            )

        constant_energy, grouped_terms = self._grouping_for_hamiltonian(hamiltonian_data)

        if not grouped_terms:
            plan = TianyanMeasurementPlan(
                circuits=(),
                groups=(),
                constant_energy=float(constant_energy),
                logical_to_physical=self.logical_to_physical,
                n_qubits=self.n_qubits,
            )
            self.last_plan = plan
            return plan

        with global_tracker.phase("Estimator: Tianyan QCIS Assembly"):
            # Tianyan superconducting hardware accepts its native QCIS basis,
            # not generic H/RX/CX instructions. Compile the logical VQE circuit
            # before physical-qubit remapping and cloud submission.
            compiled = compile_circuit(
                base_circuit,
                target_basis=_TIANYAN_NATIVE_BASIS,
            ).circuit
            base_qcis = qcis.dumps(compiled).strip()
            if _MEASUREMENT_LINE.search(base_qcis):
                raise ValueError(
                    "The VQE base circuit already contains measurement instructions. "
                    "TianyanEnergyEstimator appends grouped measurements itself."
                )
            base_qcis = self._remap_qcis(base_qcis)

            scripts: list[str] = []
            plan_groups: list[TianyanMeasurementGroup] = []
            for master, terms in grouped_terms:
                support = tuple(
                    logical
                    for logical, char in enumerate(master)
                    if char != "I"
                )
                measured_logical = (
                    support if self.measure_only_support else tuple(range(self.n_qubits))
                )
                measured_physical = tuple(
                    self.logical_to_physical[logical] for logical in measured_logical
                )

                rotations = self._rotation_qcis(master)
                measurements = self._measurement_qcis(measured_physical)
                script = "\n".join(
                    part for part in (base_qcis, rotations, measurements) if part
                )
                self._validate_native_qcis(script)
                scripts.append(script)
                plan_groups.append(
                    TianyanMeasurementGroup(
                        master_pauli=master,
                        terms=tuple(terms),
                        measured_logical_qubits=measured_logical,
                        measured_physical_qubits=measured_physical,
                    )
                )

        plan = TianyanMeasurementPlan(
            circuits=tuple(scripts),
            groups=tuple(plan_groups),
            constant_energy=float(constant_energy),
            logical_to_physical=self.logical_to_physical,
            n_qubits=self.n_qubits,
        )
        self.last_plan = plan
        return plan

    def submit(
        self, base_circuit: Circuit, hamiltonian_data: HamiltonianData
    ) -> TianyanSubmittedEvaluation:
        """Prepare and submit one grouped energy evaluation without waiting."""

        plan = self.prepare(base_circuit, hamiltonian_data)
        if not plan.circuits:
            self.last_task = None
            return TianyanSubmittedEvaluation(plan=plan, task=None)

        with global_tracker.phase("Backend: Tianyan Submit"):
            task = self._run_with_mode(list(plan.circuits))
        task_ids = tuple(str(task_id) for task_id in task.task_ids)
        if len(task_ids) != len(plan.circuits):
            raise RuntimeError(
                "Tianyan returned a task-id count that does not match the number "
                f"of grouped circuits: {len(task_ids)} != {len(plan.circuits)}"
            )
        self.last_task = task
        return TianyanSubmittedEvaluation(plan=plan, task=task)

    def collect(
        self, submitted: TianyanSubmittedEvaluation
    ) -> TianyanEnergyResult:
        """Wait for a submitted grouped evaluation and calculate its energy."""

        plan = submitted.plan
        task = submitted.task
        if task is None:
            result = TianyanEnergyResult(
                energy=float(plan.constant_energy),
                constant_energy=float(plan.constant_energy),
                groups=(),
                task_ids=(),
                shots=self.shots,
                device_name=self.device_name,
                calibration_mode=self.calibration_mode,
            )
            self.last_result = result
            return result

        with global_tracker.phase("Backend: Tianyan Wait"):
            raw_results = list(
                task.wait(
                    timeout_secs=self.timeout_secs,
                    poll_interval_secs=self.poll_interval_secs,
                )
            )

        task_ids = tuple(str(task_id) for task_id in task.task_ids)
        if len(raw_results) != len(task_ids):
            raise RuntimeError(
                "TaskHandle.wait returned a different number of results than task IDs: "
                f"{len(raw_results)} != {len(task_ids)}. Raw result order is not repaired."
            )
        if len(raw_results) != len(plan.groups):
            raise RuntimeError(
                "Tianyan result count does not match the measurement plan: "
                f"{len(raw_results)} != {len(plan.groups)}"
            )

        total = float(plan.constant_energy)
        group_results: list[TianyanGroupResult] = []
        with global_tracker.phase("Estimator: Tianyan Parity Calc"):
            for index, (group, execution_result, task_id) in enumerate(zip(
                plan.groups, raw_results, task_ids, strict=True
            )):
                result_task_id = str(execution_result.task_id)
                if result_task_id != task_id:
                    raise RuntimeError(
                        "TaskHandle.wait did not preserve submission order at index "
                        f"{index}: expected task_id={task_id!r}, got {result_task_id!r}. "
                        "The VQE layer does not reorder results."
                    )
                probabilities = self._probabilities_from_exact_counts(execution_result)
                measured_qubits = self._strict_result_qubit_indices(
                    execution_result, expected=group.measured_physical_qubits
                )
                term_results: list[TianyanTermResult] = []
                group_contribution = 0.0
                for pauli, coefficient in group.terms:
                    real_coefficient = self._real_coefficient(coefficient)
                    expectation = self._compute_physical_parity(
                        probabilities,
                        pauli,
                        measured_qubits,
                        plan.logical_to_physical,
                    )
                    contribution = real_coefficient * expectation
                    group_contribution += contribution
                    term_results.append(
                        TianyanTermResult(
                            pauli=pauli,
                            coefficient=real_coefficient,
                            expectation=float(expectation),
                            contribution=float(contribution),
                        )
                    )
                total += group_contribution
                group_results.append(
                    TianyanGroupResult(
                        master_pauli=group.master_pauli,
                        task_id=task_id,
                        measured_physical_qubits=measured_qubits,
                        terms=tuple(term_results),
                        contribution=float(group_contribution),
                    )
                )

        result = TianyanEnergyResult(
            energy=float(total),
            constant_energy=float(plan.constant_energy),
            groups=tuple(group_results),
            task_ids=task_ids,
            shots=self.shots,
            device_name=self.device_name,
            calibration_mode=self.calibration_mode,
        )
        self.last_result = result
        return result

    def evaluate_with_details(
        self, base_circuit: Circuit, hamiltonian_data: HamiltonianData
    ) -> TianyanEnergyResult:
        """Submit, wait, and return energy plus per-group diagnostics."""

        return self.collect(self.submit(base_circuit, hamiltonian_data))

    def evaluate(
        self, base_circuit: Circuit, hamiltonian_data: HamiltonianData
    ) -> float:
        """Estimator interface used by :class:`vqe.solver.VQESolver`."""

        return self.evaluate_with_details(base_circuit, hamiltonian_data).energy

    def clear_grouping_cache(self) -> None:
        """Clear the content-based Hamiltonian/QWC grouping cache."""

        self._grouping_cache = None

    def _grouping_for_hamiltonian(
        self, hamiltonian_data: HamiltonianData
    ) -> tuple[
        float, tuple[tuple[str, tuple[tuple[str, complex], ...]], ...]
    ]:
        key = tuple((pauli, complex(coefficient)) for pauli, coefficient in hamiltonian_data)
        if self._grouping_cache is not None and self._grouping_cache[0] == key:
            return self._grouping_cache[1], self._grouping_cache[2]

        identity = "I" * self.n_qubits
        constant_energy = 0.0
        sampled_terms: list[tuple[str, complex]] = []
        for pauli, coefficient in key:
            real_coefficient = self._real_coefficient(coefficient)
            if pauli == identity:
                constant_energy += real_coefficient
            else:
                sampled_terms.append((pauli, coefficient))

        if not sampled_terms:
            grouped = ()
        elif self.grouping == "qwc":
            grouped = tuple(
                (master, tuple((pauli, complex(coeff)) for pauli, coeff in terms))
                for master, terms in self.group_hamiltonian(sampled_terms)
            )
        else:
            grouped = tuple(
                (pauli, ((pauli, complex(coeff)),))
                for pauli, coeff in sampled_terms
            )
        self._grouping_cache = (key, float(constant_energy), grouped)
        return float(constant_energy), grouped

    def _run_with_mode(self, circuits: list[str]) -> TianyanTask:
        """Call the cqlib-tianyan 2.x API directly.

        There is deliberately no fallback to ``run`` or ``run_raw``. An API
        mismatch must fail at the call site so it can be fixed against the
        actual installed hardware package.
        """

        return self.backend.run_with_mode(
            circuits, self.shots, mode=self.calibration_mode
        )

    def _remap_qcis(self, source: str) -> str:
        def replace(match: re.Match[str]) -> str:
            logical = int(match.group(1))
            if logical >= self.n_qubits:
                raise ValueError(
                    f"QCIS references logical qubit Q{logical}, outside the "
                    f"configured {self.n_qubits}-qubit register"
                )
            return f"Q{self.logical_to_physical[logical]}"

        return _QUBIT_TOKEN.sub(replace, source)

    def _rotation_qcis(self, master_pauli: str) -> str:
        """Return native Tianyan basis-change instructions for measurement."""

        commands: list[str] = []
        for logical, char in enumerate(master_pauli):
            physical = self.logical_to_physical[logical]
            if char == "X":
                # H = RZ(pi) Y2P up to an irrelevant global phase.
                commands.append(f"RZ Q{physical} 3.141592653589793")
                commands.append(f"Y2P Q{physical}")
            elif char == "Y":
                # RX(+pi/2) = Y2M RZ(+pi/2) Y2P.
                commands.append(f"Y2M Q{physical}")
                commands.append(f"RZ Q{physical} 1.5707963267948966")
                commands.append(f"Y2P Q{physical}")
        return "\n".join(commands)

    @staticmethod
    def _validate_native_qcis(source: str) -> None:
        """Reject non-native gates locally instead of failing in the cloud API."""

        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            gate = line.split(maxsplit=1)[0].upper()
            if gate not in _TIANYAN_NATIVE_QCIS_GATES:
                raise RuntimeError(
                    "Tianyan QCIS contains a non-native instruction before "
                    f"submission: line {line_number}: {raw_line!r}. "
                    f"Allowed gates are {sorted(_TIANYAN_NATIVE_QCIS_GATES)}."
                )

    @staticmethod
    def _measurement_qcis(physical_qubits: Sequence[int]) -> str:
        # Separate lines match the Tianyan examples and preserve header order.
        return "\n".join(f"M Q{qubit}" for qubit in physical_qubits)

    @staticmethod
    def _probabilities_from_exact_counts(result: Any) -> Mapping[str, float]:
        """Convert the SDK's exact ``counts`` field without using fallbacks.

        ``probabilities`` is intentionally ignored. In cqlib-tianyan it may be
        built from the platform's probability-only compatibility path, where
        physical measurement headers are unavailable. That path must surface as
        a qubit-mapping mismatch instead of being silently accepted.
        """

        counts = dict(result.counts)
        if not counts:
            raise RuntimeError(
                f"Tianyan task {result.task_id!r} returned an empty counts map"
            )
        total_counts = sum(int(value) for value in counts.values())
        if total_counts <= 0:
            raise RuntimeError(
                f"Tianyan task {result.task_id!r} returned non-positive total counts"
            )

        probabilities: dict[str, float] = {}
        for bitstring, count in counts.items():
            key = str(bitstring)
            if not key or any(bit not in "01" for bit in key):
                raise RuntimeError(
                    f"Tianyan task {result.task_id!r} returned invalid bitstring {key!r}"
                )
            value = int(count)
            if value < 0:
                raise RuntimeError(
                    f"Tianyan task {result.task_id!r} returned negative count for {key!r}"
                )
            probabilities[key] = value / total_counts
        return probabilities

    @staticmethod
    def _qubit_index(qubit: Any) -> int:
        if isinstance(qubit, int):
            return int(qubit)
        if hasattr(qubit, "index"):
            return int(qubit.index)
        if hasattr(qubit, "id"):
            return int(qubit.id)
        raise TypeError(f"Cannot extract a qubit index from {qubit!r}")

    @classmethod
    def _strict_result_qubit_indices(
        cls, result: Any, *, expected: Sequence[int]
    ) -> tuple[int, ...]:
        """Require the SDK result header to exactly match emitted measurements."""

        qubits = result.qubits
        indices = tuple(cls._qubit_index(qubit) for qubit in qubits)
        expected_indices = tuple(int(qubit) for qubit in expected)
        if indices != expected_indices:
            raise RuntimeError(
                f"Tianyan task {result.task_id!r} returned result.qubits={indices}, "
                f"but the emitted QCIS measured {expected_indices}. No fallback or "
                "synthetic-qubit repair is applied."
            )
        return indices

    @staticmethod
    def _compute_physical_parity(
        probabilities: Mapping[str, float],
        pauli: str,
        measured_physical_qubits: Sequence[int],
        logical_to_physical: Sequence[int],
    ) -> float:
        """Calculate a Pauli expectation using ExecutionResult qubit ordering.

        ``ExecutionResult`` defines ``qubits[i]`` as bit ``i`` in the outcome;
        therefore bit ``i`` is printed at position ``-(i+1)`` in the big-endian
        bitstring.  Building an integer mask over the result-qubit positions
        handles arbitrary physical-qubit subsets and orderings correctly.
        """

        position = {
            int(physical): index
            for index, physical in enumerate(measured_physical_qubits)
        }
        support_positions: list[int] = []
        for logical, char in enumerate(pauli):
            if char == "I":
                continue
            physical = int(logical_to_physical[logical])
            if physical not in position:
                raise RuntimeError(
                    f"Pauli term {pauli!r} needs physical qubit Q{physical}, "
                    "but it was not present in the Tianyan result"
                )
            support_positions.append(position[physical])

        width = len(measured_physical_qubits)
        mask = sum(1 << bit_position for bit_position in support_positions)
        expectation = 0.0
        for bitstring, probability in probabilities.items():
            if len(bitstring) != width:
                raise RuntimeError(
                    f"Tianyan bitstring {bitstring!r} has width {len(bitstring)}, "
                    f"expected {width} from result.qubits"
                )
            parity = (int(bitstring, 2) & mask).bit_count() & 1
            expectation += float(probability) * (-1.0 if parity else 1.0)
        return float(expectation)


# Descriptive alias retained for users migrating from CloudEnergyEstimator.
TianyanCloudEnergyEstimator = TianyanEnergyEstimator


__all__ = [
    "TianyanMeasurementGroup",
    "TianyanMeasurementPlan",
    "TianyanTermResult",
    "TianyanGroupResult",
    "TianyanEnergyResult",
    "TianyanSubmittedEvaluation",
    "TianyanEnergyEstimator",
    "TianyanCloudEnergyEstimator",
]
