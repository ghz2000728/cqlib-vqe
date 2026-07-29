"""Canonical closed-shell singlet-UCCSD parameter conventions.

The crucial correctness rule is that classical CCSD amplitudes and quantum
excitation generators use the same OpenFermion packed singlet basis.  A packed
index identifies exactly one physical variational parameter and one one-hot
anti-Hermitian generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SingletExcitation:
    kind: str
    pair_a: tuple[int, int]
    pair_b: tuple[int, int] | None = None
    packed_index: int = -1


@dataclass(frozen=True)
class CanonicalBasisAudit:
    relative_residual: float
    absolute_residual: float
    target_norm: float
    rank: int
    parameter_count: int
    term_count: int
    audited: bool = True


ProjectionReport = CanonicalBasisAudit


def closed_shell_dimensions(n_qubits: int, n_electrons: int) -> tuple[int, int, int]:
    if n_qubits <= 0 or n_qubits % 2:
        raise ValueError("Closed-shell singlet UCCSD requires a positive even n_qubits")
    if n_electrons <= 0 or n_electrons % 2:
        raise ValueError("Closed-shell singlet UCCSD requires a positive even electron count")
    n_spatial = n_qubits // 2
    n_occupied = n_electrons // 2
    if n_occupied > n_spatial:
        raise ValueError("Electron count exceeds the available spin orbitals")
    return n_spatial, n_occupied, n_spatial - n_occupied


def canonical_parameter_count(n_qubits: int, n_electrons: int) -> int:
    n_spatial, n_occupied, n_virtual = closed_shell_dimensions(n_qubits, n_electrons)
    del n_spatial
    n_pairs = n_occupied * n_virtual
    return 2 * n_pairs + n_pairs * (n_pairs - 1) // 2


def canonical_singlet_descriptors(
    n_qubits: int,
    n_electrons: int,
) -> list[SingletExcitation]:
    """Return descriptors in OpenFermion's packed singlet-UCCSD order."""

    n_spatial, n_occupied, _ = closed_shell_dimensions(n_qubits, n_electrons)
    pairs = [
        (occupied, virtual)
        for virtual in range(n_occupied, n_spatial)
        for occupied in range(n_occupied)
    ]
    descriptors: list[SingletExcitation] = []
    for packed_index, pair in enumerate(pairs):
        descriptors.append(SingletExcitation("single", pair, None, packed_index))
    offset = len(pairs)
    for local_index, pair in enumerate(pairs):
        descriptors.append(
            SingletExcitation("double", pair, pair, offset + local_index)
        )
    offset += len(pairs)
    for local_index, (pair_a, pair_b) in enumerate(combinations(pairs, 2)):
        descriptors.append(
            SingletExcitation("double", pair_a, pair_b, offset + local_index)
        )
    expected = canonical_parameter_count(n_qubits, n_electrons)
    if len(descriptors) != expected:
        raise RuntimeError(f"Canonical descriptor count {len(descriptors)} != {expected}")
    return descriptors


def canonical_singlet_operator(packed_index: int, n_qubits: int, n_electrons: int):
    """Return one one-hot anti-Hermitian OpenFermion singlet generator."""

    try:
        from openfermion import normal_ordered, uccsd_singlet_generator
    except ImportError as exc:
        raise ImportError("canonical singlet generators require openfermion") from exc

    parameter_count = canonical_parameter_count(n_qubits, n_electrons)
    if not 0 <= packed_index < parameter_count:
        raise ValueError(
            f"Invalid packed index {packed_index}; expected [0, {parameter_count})"
        )
    one_hot = np.zeros(parameter_count, dtype=np.float64)
    one_hot[packed_index] = 1.0
    return normal_ordered(
        uccsd_singlet_generator(
            one_hot.tolist(), n_qubits, n_electrons, anti_hermitian=True
        )
    )


def build_singlet_basis(n_qubits: int, n_electrons: int):
    descriptors = canonical_singlet_descriptors(n_qubits, n_electrons)
    operators = [
        canonical_singlet_operator(item.packed_index, n_qubits, n_electrons)
        for item in descriptors
    ]
    return descriptors, operators


def _normalized_terms(operator, atol: float) -> dict[tuple, complex]:
    from openfermion import normal_ordered

    normalized = normal_ordered(operator)
    normalized.compress(abs_tol=atol)
    return {term: complex(value) for term, value in normalized.terms.items()}


def audit_fermion_operator(target, basis: Sequence, *, atol: float = 1e-12):
    if not basis:
        raise ValueError("The audit basis cannot be empty")
    target_terms = _normalized_terms(target, atol)
    basis_terms = [_normalized_terms(operator, atol) for operator in basis]
    all_terms = sorted(
        set(target_terms).union(*(set(terms) for terms in basis_terms)), key=repr
    )
    row = {term: index for index, term in enumerate(all_terms)}
    matrix = np.zeros((len(all_terms), len(basis)), dtype=np.complex128)
    vector = np.zeros(len(all_terms), dtype=np.complex128)
    for term, value in target_terms.items():
        vector[row[term]] = value
    for column, terms in enumerate(basis_terms):
        for term, value in terms.items():
            matrix[row[term], column] = value
    parameters, _, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
    fitted = matrix @ parameters
    absolute_residual = float(np.linalg.norm(fitted - vector))
    target_norm = float(np.linalg.norm(vector))
    relative_residual = absolute_residual / max(target_norm, np.finfo(float).eps)
    imaginary_norm = float(np.linalg.norm(parameters.imag))
    real_norm = float(np.linalg.norm(parameters.real))
    if imaginary_norm > atol * max(1.0, real_norm):
        raise ValueError(
            f"Canonical audit returned complex parameters: ||Im||={imaginary_norm:.3e}"
        )
    return np.asarray(parameters.real, dtype=np.float64), CanonicalBasisAudit(
        relative_residual=relative_residual,
        absolute_residual=absolute_residual,
        target_norm=target_norm,
        rank=int(rank),
        parameter_count=len(basis),
        term_count=len(all_terms),
        audited=True,
    )


def pauli_strings_commute(left: str, right: str) -> bool:
    if len(left) != len(right):
        raise ValueError("Pauli strings must have equal width")
    anti_sites = sum(
        1
        for lhs, rhs in zip(left, right)
        if lhs != "I" and rhs != "I" and lhs != rhs
    )
    return anti_sites % 2 == 0


def pauli_data_pairwise_commute(pauli_data: Sequence[tuple[str, float]]) -> bool:
    strings = [item[0] for item in pauli_data]
    return all(
        pauli_strings_commute(strings[i], strings[j])
        for i in range(len(strings))
        for j in range(i + 1, len(strings))
    )


project_fermion_operator = audit_fermion_operator

__all__ = [
    "CanonicalBasisAudit",
    "ProjectionReport",
    "SingletExcitation",
    "audit_fermion_operator",
    "build_singlet_basis",
    "canonical_parameter_count",
    "canonical_singlet_descriptors",
    "canonical_singlet_operator",
    "closed_shell_dimensions",
    "pauli_data_pairwise_commute",
    "pauli_strings_commute",
    "project_fermion_operator",
]
