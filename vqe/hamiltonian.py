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

"""Hamiltonian conversion helpers for the cqlib2 QIS API.

The legacy VQE project stores Pauli strings in ``q0-left`` order: character
``i`` acts on qubit ``i``.  cqlib2's :class:`PauliString.from_str` follows the
usual display convention where the rightmost character acts on qubit 0.
This module is the single boundary that performs that reversal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number
from typing import Sequence, TypeAlias

from cqlib.qis import Hamiltonian, PauliString

Coefficient: TypeAlias = complex | float | int
HamiltonianTerm: TypeAlias = tuple[str, Coefficient]
HamiltonianData: TypeAlias = Sequence[HamiltonianTerm]


def validate_hamiltonian_data(
    hamiltonian_data: HamiltonianData,
    *,
    expected_num_qubits: int | None = None,
) -> int:
    """Validate the legacy list representation and return its qubit count."""
    if not hamiltonian_data:
        raise ValueError("hamiltonian_data must contain at least one Pauli term")

    first_string = hamiltonian_data[0][0]
    if not isinstance(first_string, str) or not first_string:
        raise ValueError("Pauli strings must be non-empty strings")
    num_qubits = len(first_string)

    if expected_num_qubits is not None and num_qubits != expected_num_qubits:
        raise ValueError(
            f"Hamiltonian has {num_qubits} qubits, expected {expected_num_qubits}"
        )

    for index, item in enumerate(hamiltonian_data):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"Hamiltonian term {index} must be a (pauli, coeff) pair")
        pauli, coeff = item
        if not isinstance(pauli, str) or len(pauli) != num_qubits:
            raise ValueError(
                f"Hamiltonian term {index} has invalid Pauli length; expected {num_qubits}"
            )
        if any(char not in "IXYZ" for char in pauli):
            raise ValueError(f"Hamiltonian term {index} contains an invalid Pauli character")
        if not isinstance(coeff, Number):
            raise TypeError(f"Hamiltonian coefficient {index} is not numeric")
        value = complex(coeff)
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError(f"Hamiltonian coefficient {index} must be finite")

    return num_qubits


def legacy_to_cqlib_pauli(pauli_q0_left: str) -> PauliString:
    """Convert a legacy ``q0-left`` string into a cqlib2 PauliString."""
    if not pauli_q0_left or any(char not in "IXYZ" for char in pauli_q0_left):
        raise ValueError(f"Invalid Pauli string: {pauli_q0_left!r}")
    return PauliString.from_str(pauli_q0_left[::-1])


def build_cqlib_hamiltonian(
    hamiltonian_data: HamiltonianData,
    *,
    expected_num_qubits: int | None = None,
) -> Hamiltonian:
    """Build and simplify a native cqlib2 Hamiltonian."""
    num_qubits = validate_hamiltonian_data(
        hamiltonian_data, expected_num_qubits=expected_num_qubits
    )
    native = Hamiltonian(num_qubits)
    for pauli, coeff in hamiltonian_data:
        native.add_term(legacy_to_cqlib_pauli(pauli), complex(coeff))
    native.simplify()
    return native


def hamiltonian_cache_key(hamiltonian_data: HamiltonianData) -> tuple[tuple[str, complex], ...]:
    """Return a content-based cache key."""
    validate_hamiltonian_data(hamiltonian_data)
    return tuple((pauli, complex(coeff)) for pauli, coeff in hamiltonian_data)


@dataclass(slots=True)
class NativeHamiltonianCache:
    """Single-entry cache optimized for repeated VQE objective evaluations."""

    _key: tuple[tuple[str, complex], ...] | None = None
    _value: Hamiltonian | None = None

    def get(
        self,
        hamiltonian_data: HamiltonianData,
        *,
        expected_num_qubits: int | None = None,
    ) -> Hamiltonian:
        key = hamiltonian_cache_key(hamiltonian_data)
        if key != self._key or self._value is None:
            self._value = build_cqlib_hamiltonian(
                hamiltonian_data, expected_num_qubits=expected_num_qubits
            )
            self._key = key
        elif expected_num_qubits is not None and self._value.num_qubits != expected_num_qubits:
            raise ValueError(
                f"Cached Hamiltonian has {self._value.num_qubits} qubits, "
                f"expected {expected_num_qubits}"
            )
        return self._value

    def clear(self) -> None:
        self._key = None
        self._value = None
