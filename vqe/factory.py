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

"""Compiled UCCSD circuit factory for the cqlib2 dynamic API.

The OpenFermion mapping is performed once during factory construction. Native
``Statevector.apply_pauli_rotation`` execution is used when available; other
bindings use the decomposed gate path. Three circuit construction modes are
supported:

``jit``
    Rebuild a numeric :class:`cqlib.Circuit` from the cached gate plan.
``bind``
    Build one symbolic cqlib2 circuit and call ``assign_parameters``.
``direct_statevector``
    Apply the cached gate plan directly to :class:`cqlib.qis.Statevector`,
    avoiding both circuit reconstruction and parameter binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np

from cqlib import Circuit, Parameter
from cqlib.qis import PauliString, Statevector

from utils.profiler import global_tracker


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PauliRotationTerm:
    """One Pauli term in an anti-Hermitian excitation generator.

    ``pauli`` uses the legacy internal convention: character ``i`` acts on
    qubit ``i``. ``coefficient`` is the imaginary coefficient ``c`` in
    ``G = i * c * P``. The implemented Pauli rotation angle is ``-2 θ c``.
    """

    pauli: str
    coefficient: float


@dataclass(frozen=True, slots=True)
class CompiledGenerator:
    """Cached qubit-operator expansion for one variational parameter."""

    terms: tuple[PauliRotationTerm, ...]
    label: str = ""


@dataclass(frozen=True, slots=True)
class CompiledStatevectorRotation:
    """One precompiled statevector operation for a variational parameter."""

    angle_scale: float
    active_paulis: tuple[tuple[int, str], ...]
    native_pauli: PauliString | None


class UCCSDFactory:
    """Spin-adapted selected-UCCSD factory for JW, BK, and parity mappings.

    The input ``molecule_data`` is expected to expose the fields produced by
    :class:`chemistry.molecule.MolecularDataEngine`.
    """

    SUPPORTED_MODES = {"jit", "bind"}

    def __init__(
        self,
        molecule_data,
        *,
        construction_mode: str = "jit",
        coefficient_threshold: float = 1e-12,
        trotter_steps: int = 1,
        trotter_order: int = 2,
        selected_packed_indices: Sequence[int] | None = None,
    ) -> None:
        from chemistry.mapping import encoded_occupied_qubits, normalize_mapper_type

        self.mapper_type = normalize_mapper_type(getattr(molecule_data, "mapper_type", "jw"))

        self.n_qubits = int(molecule_data.n_qubits)
        self.n_electrons = int(molecule_data.n_electrons)
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if not 0 <= self.n_electrons <= self.n_qubits:
            raise ValueError("n_electrons must be between 0 and n_qubits")
        if selected_packed_indices is None:
            self.singles = list(molecule_data.sp_singles)
            self.doubles = list(molecule_data.sp_doubles)
        else:
            if not hasattr(molecule_data, "ccsd_items"):
                raise TypeError(
                    "selected_packed_indices requires molecule_data.ccsd_items(...)"
                )
            requested = tuple(sorted({int(value) for value in selected_packed_indices}))
            self.singles, self.doubles = molecule_data.ccsd_items(requested)
        selected_items = self.singles + self.doubles
        if selected_items and not all("packed_index" in item for item in selected_items):
            raise ValueError(
                "Closed-shell UCCSD requires canonical packed_index metadata. "
                "Rebuild the molecular data with canonical UCCSD descriptors."
            )
        self.selected_items = sorted(
            selected_items, key=lambda item: int(item["packed_index"])
        )
        packed_indices = [int(item["packed_index"]) for item in self.selected_items]
        if len(packed_indices) != len(set(packed_indices)):
            raise ValueError("Canonical UCCSD metadata contains duplicate packed indices")
        self.selected_packed_indices = tuple(packed_indices)
        self.full_parameter_count = int(
            getattr(molecule_data, "ccsd_full_parameter_count", len(self.selected_items))
        )
        self.coefficient_threshold = float(coefficient_threshold)
        self.trotter_steps = int(trotter_steps)
        self.trotter_order = int(trotter_order)
        if self.trotter_steps <= 0:
            raise ValueError("trotter_steps must be positive")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        self.construction_mode = self._validate_mode(construction_mode)
        self.occupied_qubits = encoded_occupied_qubits(
            n_spin_orbitals=self.n_qubits,
            occupied_spin_orbitals=range(self.n_electrons),
            mapper_type=self.mapper_type,
        )
        self._validate_occupied_qubits()

        self.initial_values = np.asarray(
            [item["amplitude"] for item in self.selected_items], dtype=float
        )

        with global_tracker.phase("Factory: Compile Generators"):
            self.generators = self._compile_excitation_generators()

        self._native_pauli_rotation_available = callable(
            getattr(Statevector, "apply_pauli_rotation", None)
        )
        self._native_pauli_cache: dict[str, PauliString] = {}
        self._statevector_plans = self._compile_statevector_plans()

        self.num_params = len(self.generators)
        if self.initial_values.size != self.num_params:
            raise RuntimeError(
                f"Compiled parameter count mismatch: {self.num_params} generators but "
                f"{self.initial_values.size} initial values"
            )

        self.parameter_names = tuple(f"theta_{index}" for index in range(self.num_params))
        self._template: Circuit | None = None
        if self.construction_mode == "bind":
            with global_tracker.phase("Factory: Build Symbolic Template"):
                self._template = self._build_symbolic_template()

    @classmethod
    def from_compiled(
        cls,
        *,
        n_qubits: int,
        n_electrons: int,
        generators: Sequence[Sequence[tuple[str, float] | PauliRotationTerm]],
        initial_values: Sequence[float] | None = None,
        construction_mode: str = "jit",
        occupied_qubits: Sequence[int] | None = None,
        trotter_steps: int = 1,
        trotter_order: int = 1,
    ) -> "UCCSDFactory":
        """Create a factory from a precompiled gate plan.

        This is useful for tests, serialization, and deployments where
        OpenFermion is intentionally absent from the runtime environment.
        """
        self = cls.__new__(cls)
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if not 0 <= self.n_electrons <= self.n_qubits:
            raise ValueError("n_electrons must be between 0 and n_qubits")
        self.singles = []
        self.doubles = []
        self.mapper_type = "compiled"
        self.selected_items = []
        self.selected_packed_indices = ()
        self.full_parameter_count = len(generators)
        self.coefficient_threshold = 0.0
        self.trotter_steps = int(trotter_steps)
        self.trotter_order = int(trotter_order)
        if self.trotter_steps <= 0:
            raise ValueError("trotter_steps must be positive")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        self.construction_mode = self._validate_mode(construction_mode)
        self.occupied_qubits = tuple(
            range(self.n_electrons) if occupied_qubits is None else occupied_qubits
        )
        self._validate_occupied_qubits()

        compiled: list[CompiledGenerator] = []
        for generator_index, generator in enumerate(generators):
            terms: list[PauliRotationTerm] = []
            for term in generator:
                if isinstance(term, PauliRotationTerm):
                    parsed = term
                else:
                    parsed = PauliRotationTerm(str(term[0]), float(term[1]))
                self._validate_pauli(parsed.pauli)
                terms.append(parsed)
            compiled.append(CompiledGenerator(tuple(terms), f"compiled_{generator_index}"))
        self.generators = tuple(compiled)
        self._native_pauli_rotation_available = callable(
            getattr(Statevector, "apply_pauli_rotation", None)
        )
        self._native_pauli_cache = {}
        self._statevector_plans = self._compile_statevector_plans()
        self.num_params = len(self.generators)
        self.initial_values = np.asarray(
            np.zeros(self.num_params) if initial_values is None else initial_values,
            dtype=float,
        )
        if self.initial_values.size != self.num_params:
            raise ValueError("initial_values length must match the number of generators")
        self.parameter_names = tuple(f"theta_{index}" for index in range(self.num_params))
        self._template = self._build_symbolic_template() if self.construction_mode == "bind" else None
        return self

    def _validate_occupied_qubits(self) -> None:
        if len(set(self.occupied_qubits)) != len(self.occupied_qubits):
            raise ValueError("occupied_qubits must be unique")
        if any(not 0 <= int(qubit) < self.n_qubits for qubit in self.occupied_qubits):
            raise ValueError("occupied_qubits contains an out-of-range qubit")
        # In Jordan-Wigner the Hamming weight equals the electron count.  BK
        # and parity encode occupation through an invertible binary transform,
        # so the reference computational state can have a different Hamming
        # weight while still representing the same fermionic determinant.
        if self.mapper_type == "jw" and len(self.occupied_qubits) != self.n_electrons:
            raise ValueError("occupied_qubits length must equal n_electrons")

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = str(mode).lower()
        if normalized not in UCCSDFactory.SUPPORTED_MODES:
            raise ValueError(
                f"construction_mode must be one of {sorted(UCCSDFactory.SUPPORTED_MODES)}, "
                f"got {mode!r}"
            )
        return normalized

    def set_construction_mode(self, mode: str) -> None:
        """Switch between numeric JIT construction and symbolic binding."""
        self.construction_mode = self._validate_mode(mode)
        if self.construction_mode == "bind" and self._template is None:
            with global_tracker.phase("Factory: Build Symbolic Template"):
                self._template = self._build_symbolic_template()

    def _compile_excitation_generators(self) -> tuple[CompiledGenerator, ...]:
        try:
            from chemistry.mapping import map_fermion_operator
            from chemistry.uccsd import (
                canonical_singlet_operator,
                pauli_data_pairwise_commute,
            )
        except ImportError as exc:
            raise ImportError(
                "Canonical UCCSD generator compilation requires openfermion"
            ) from exc

        generators: list[CompiledGenerator] = []
        noncommuting: list[int] = []
        for item in self.selected_items:
            packed_index = int(item["packed_index"])
            fermion_generator = canonical_singlet_operator(
                packed_index, self.n_qubits, self.n_electrons
            )
            compiled = self._compile_qubit_generator(
                map_fermion_operator(
                    fermion_generator,
                    mapper_type=self.mapper_type,
                    n_spin_orbitals=self.n_qubits,
                ),
                label=f"canonical_{packed_index}",
            )
            if not pauli_data_pairwise_commute(
                [(term.pauli, term.coefficient) for term in compiled.terms]
            ):
                noncommuting.append(packed_index)
            generators.append(compiled)

        self.noncommuting_packed_indices = tuple(noncommuting)
        if noncommuting:
            _LOGGER.debug(
                "Canonical singlet generators contain noncommuting Pauli terms at "
                "packed indices %s; using order-%d product formula with %d step(s).",
                noncommuting,
                self.trotter_order,
                self.trotter_steps,
            )
        return tuple(generators)

    def _compile_qubit_generator(self, qubit_operator, *, label: str) -> CompiledGenerator:
        terms: list[PauliRotationTerm] = []
        for term, coefficient in qubit_operator.terms.items():
            if abs(float(np.real(coefficient))) > 1e-10:
                raise ValueError(
                    f"Excitation generator {label} is not anti-Hermitian after mapping: "
                    f"coefficient {coefficient!r} has a real component"
                )
            imag_coeff = float(np.imag(coefficient))
            if abs(imag_coeff) <= self.coefficient_threshold:
                continue
            pauli_chars = ["I"] * self.n_qubits
            for qubit, char in term:
                pauli_chars[int(qubit)] = str(char)
            pauli = "".join(pauli_chars)
            self._validate_pauli(pauli)
            terms.append(PauliRotationTerm(pauli, imag_coeff))
        return CompiledGenerator(tuple(terms), label)

    def _validate_pauli(self, pauli: str) -> None:
        if len(pauli) != self.n_qubits or any(char not in "IXYZ" for char in pauli):
            raise ValueError(
                f"Invalid {self.n_qubits}-qubit Pauli string in compiled generator: {pauli!r}"
            )

    def build(self, parameter_values: Sequence[float]) -> Circuit:
        """Build a numeric circuit using the selected construction mode."""
        params = self._coerce_parameters(parameter_values)
        if self.construction_mode == "bind":
            if self._template is None:
                self._template = self._build_symbolic_template()
            bindings = dict(zip(self.parameter_names, params, strict=True))
            with global_tracker.phase("Factory: Assign Parameters"):
                return self._template.assign_parameters(bindings)

        with global_tracker.phase("Factory: Build Circuit"):
            circuit = Circuit(self.n_qubits)
            self._prepare_reference_circuit(circuit)
            for theta, generator in zip(params, self.generators, strict=True):
                self._apply_generator_to_circuit(circuit, generator, float(theta))
            return circuit

    @property
    def native_pauli_rotation_available(self) -> bool:
        """Whether the installed cqlib2 exposes the fused Pauli kernel."""
        return bool(self._native_pauli_rotation_available)

    def build_statevector(self, parameter_values: Sequence[float]) -> Statevector:
        """Evolve a native Statevector using a precompiled rotation plan.

        cqlib2 builds with ``Statevector.apply_pauli_rotation`` execute each
        Pauli exponential as one Rust call. Older cqlib2 builds transparently
        fall back to the 1.3.x basis-change/CNOT decomposition.
        """
        params = self._coerce_parameters(parameter_values)
        with global_tracker.phase("Factory: Direct Statevector"):
            state = Statevector(self.n_qubits)
            for qubit in self.occupied_qubits:
                state.apply_x(int(qubit))
            for theta, plan in zip(params, self._statevector_plans, strict=True):
                self._apply_statevector_plan(state, plan, float(theta))
            return state

    def _build_symbolic_template(self) -> Circuit:
        circuit = Circuit(self.n_qubits)
        self._prepare_reference_circuit(circuit)
        for name, generator in zip(self.parameter_names, self.generators, strict=True):
            theta = Parameter(name)
            self._apply_generator_to_circuit(circuit, generator, theta)
        return circuit

    def _prepare_reference_circuit(self, circuit: Circuit) -> None:
        for qubit in self.occupied_qubits:
            circuit.x(int(qubit))

    def _coerce_parameters(self, parameter_values: Sequence[float]) -> np.ndarray:
        params = np.asarray(parameter_values, dtype=float).reshape(-1)
        if params.size != self.num_params:
            raise ValueError(
                f"Parameter mismatch: expected {self.num_params}, got {params.size}"
            )
        if not np.all(np.isfinite(params)):
            raise ValueError("All variational parameters must be finite")
        return params

    @staticmethod
    def _rotation_angle(theta, coefficient: float):
        # exp(theta * i*c*P) = exp(-i * angle/2 * P), angle = -2*theta*c.
        return (-2.0 * coefficient) * theta

    def _trotter_schedule(
        self, generator: CompiledGenerator
    ) -> tuple[tuple[PauliRotationTerm, float], ...]:
        if self.trotter_order == 1 or len(generator.terms) <= 1:
            return tuple((term, 1.0) for term in generator.terms)
        forward = tuple((term, 0.5) for term in generator.terms)
        backward = tuple((term, 0.5) for term in reversed(generator.terms))
        return forward + backward

    def _apply_generator_to_circuit(
        self, circuit: Circuit, generator: CompiledGenerator, theta
    ) -> None:
        schedule = self._trotter_schedule(generator)
        for _ in range(self.trotter_steps):
            for term, weight in schedule:
                self._apply_pauli_rotation_circuit(
                    circuit,
                    term.pauli,
                    self._rotation_angle(
                        theta, weight * term.coefficient / self.trotter_steps
                    ),
                )

    def _compile_native_pauli(self, pauli: str) -> PauliString | None:
        if not self._native_pauli_rotation_available:
            return None
        cached = self._native_pauli_cache.get(pauli)
        if cached is None:
            # Legacy strings are q0-left; cqlib PauliString display is q0-right.
            cached = PauliString.from_str(pauli[::-1])
            self._native_pauli_cache[pauli] = cached
        return cached

    def _compile_statevector_plans(
        self,
    ) -> tuple[tuple[CompiledStatevectorRotation, ...], ...]:
        plans: list[tuple[CompiledStatevectorRotation, ...]] = []
        for generator in self.generators:
            rotations: list[CompiledStatevectorRotation] = []
            schedule = self._trotter_schedule(generator)
            for _ in range(self.trotter_steps):
                for term, weight in schedule:
                    rotations.append(
                        CompiledStatevectorRotation(
                            angle_scale=float(
                                -2.0 * weight * term.coefficient / self.trotter_steps
                            ),
                            active_paulis=tuple(self._active_paulis(term.pauli)),
                            native_pauli=self._compile_native_pauli(term.pauli),
                        )
                    )
            plans.append(tuple(rotations))
        return tuple(plans)

    def _apply_statevector_plan(
        self,
        state: Statevector,
        plan: Sequence[CompiledStatevectorRotation],
        theta: float,
    ) -> None:
        if abs(theta) < 1e-15:
            return
        for rotation in plan:
            angle = rotation.angle_scale * theta
            if abs(angle) < 1e-15:
                continue
            if rotation.native_pauli is not None:
                state.apply_pauli_rotation(rotation.native_pauli, angle)
            else:
                self._apply_pauli_rotation_statevector_decomposed(
                    state, rotation.active_paulis, angle
                )

    @staticmethod
    def _active_paulis(pauli: str) -> list[tuple[int, str]]:
        return [(index, char) for index, char in enumerate(pauli) if char != "I"]

    def _apply_pauli_rotation_circuit(self, circuit: Circuit, pauli: str, angle) -> None:
        active = self._active_paulis(pauli)
        if not active:
            return
        for qubit, char in active:
            if char == "X":
                circuit.h(qubit)
            elif char == "Y":
                circuit.rx(qubit, np.pi / 2)

        qubits = [qubit for qubit, _ in active]
        for control, target in zip(qubits[:-1], qubits[1:]):
            circuit.cx(control, target)
        circuit.rz(qubits[-1], angle)
        for control, target in reversed(tuple(zip(qubits[:-1], qubits[1:]))):
            circuit.cx(control, target)

        for qubit, char in active:
            if char == "X":
                circuit.h(qubit)
            elif char == "Y":
                circuit.rx(qubit, -np.pi / 2)

    @staticmethod
    def _apply_pauli_rotation_statevector_decomposed(
        state: Statevector, active: Sequence[tuple[int, str]], angle: float
    ) -> None:
        if not active:
            return
        for qubit, char in active:
            if char == "X":
                state.apply_h(qubit)
            elif char == "Y":
                state.apply_rx(qubit, np.pi / 2)

        qubits = tuple(qubit for qubit, _ in active)
        chain = tuple(zip(qubits[:-1], qubits[1:]))
        for control, target in chain:
            state.apply_cx(control, target)
        state.apply_rz(qubits[-1], angle)
        for control, target in reversed(chain):
            state.apply_cx(control, target)

        for qubit, char in active:
            if char == "X":
                state.apply_h(qubit)
            elif char == "Y":
                state.apply_rx(qubit, -np.pi / 2)


# Backward-compatible class name used by the original project.
UCCSD_Factory = UCCSDFactory

__all__ = [
    "PauliRotationTerm",
    "CompiledGenerator",
    "CompiledStatevectorRotation",
    "UCCSDFactory",
    "UCCSD_Factory",
]
