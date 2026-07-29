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

"""Shared fermion-to-qubit mapping and reference-state helpers.

The UCCSD ansatz and the molecular Hamiltonian must be transformed by the
same encoding.  For non-Jordan-Wigner encodings, the Hartree-Fock occupation
vector must also be encoded before preparing the computational-basis reference
state.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np


MapperType = Literal["jw", "bk", "parity"]


_ALIASES = {
    "jw": "jw",
    "jordan_wigner": "jw",
    "jordan-wigner": "jw",
    "bk": "bk",
    "bravyi_kitaev": "bk",
    "bravyi-kitaev": "bk",
    "parity": "parity",
}


def _binary_code_transform_compat(fermion_operator, code):
    """OpenFermion binary-code transform with NumPy-2-safe scalar coercion.

    OpenFermion 1.6 can pass a ``numpy.int64`` coefficient to
    ``QubitOperator`` in the parity-sign step.  Some supported OpenFermion
    releases reject that coefficient type.  This is the upstream algorithm
    with the parity exponent converted to the Python ``int`` expected by the
    operator constructor.
    """

    from openfermion import BinaryCode, FermionOperator, QubitOperator
    from openfermion.ops.operators import BinaryPolynomial
    from openfermion.transforms.opconversions.binary_code_transform import (
        extractor,
        make_parity_list,
    )

    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError("binary-code mapping requires a FermionOperator")
    if not isinstance(code, BinaryCode):
        raise TypeError("binary-code mapping requires an OpenFermion BinaryCode")

    mapped = QubitOperator()
    parity_list = make_parity_list(code)
    for term, coefficient in fermion_operator.terms.items():
        updated_parity = 0
        parity_term = BinaryPolynomial()
        changed_occupation = [0] * code.n_modes
        transformed = QubitOperator(())
        fermionic_indices = np.array([], dtype=int)
        for op_index, op_tuple in enumerate(reversed(term)):
            mode, action = int(op_tuple[0]), int(op_tuple[1])
            fermionic_indices = np.append(fermionic_indices, mode)
            count = int(np.count_nonzero(fermionic_indices[:op_index] == mode))
            updated_parity += int(
                np.count_nonzero(fermionic_indices[:op_index] < mode)
            )
            extracted = extractor(code.decoder[mode])
            extracted *= ((-1) ** count) * ((-1) ** action) * 0.5
            transformed *= QubitOperator((), 0.5) - extracted
            changed_occupation[mode] += 1
            parity_term += parity_list[mode]

        transformed *= QubitOperator((), int((-1) ** updated_parity))
        transformed *= extractor(parity_term)
        changed_qubits = np.mod(code.encoder.dot(changed_occupation), 2)
        update = QubitOperator(())
        for index, value in enumerate(changed_qubits):
            if int(value):
                update *= QubitOperator("X" + str(index))
        mapped += coefficient * update * transformed
    mapped.compress()
    return mapped


def normalize_mapper_type(mapper_type: str) -> MapperType:
    """Return the canonical mapper name accepted by the VQE pipeline."""

    normalized = str(mapper_type).lower().replace("-", "_")
    try:
        return _ALIASES[normalized]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            "mapper_type must be 'jw', 'bk', or 'parity' "
            "(aliases jordan_wigner and bravyi_kitaev are accepted)"
        ) from exc


def map_fermion_operator(fermion_operator, *, mapper_type: str, n_spin_orbitals: int):
    """Map a FermionOperator with the requested encoding.

    ``n_spin_orbitals`` is required for the parity code and is also validated
    for the other mappings to keep all ansatz and Hamiltonian paths aligned.
    """

    if int(n_spin_orbitals) <= 0:
        raise ValueError("n_spin_orbitals must be positive")
    mapper = normalize_mapper_type(mapper_type)
    if mapper == "jw":
        try:
            from openfermion import jordan_wigner
        except ImportError as exc:
            raise ImportError("Jordan-Wigner mapping requires openfermion") from exc
        return jordan_wigner(fermion_operator)
    if mapper == "bk":
        try:
            from openfermion import bravyi_kitaev
        except ImportError as exc:
            raise ImportError("Bravyi-Kitaev mapping requires openfermion") from exc
        return bravyi_kitaev(fermion_operator)
    try:
        from openfermion import FermionOperator, get_fermion_operator, parity_code
    except ImportError as exc:
        raise ImportError("parity mapping requires openfermion") from exc
    fermion = (
        fermion_operator
        if isinstance(fermion_operator, FermionOperator)
        else get_fermion_operator(fermion_operator)
    )
    return _binary_code_transform_compat(fermion, parity_code(int(n_spin_orbitals)))


def encoded_occupied_qubits(
    *,
    n_spin_orbitals: int,
    occupied_spin_orbitals: Iterable[int],
    mapper_type: str,
) -> tuple[int, ...]:
    """Encode a fermionic occupation vector as occupied computational qubits.

    Jordan-Wigner stores occupation directly.  Bravyi-Kitaev and parity use
    OpenFermion's binary encoder matrix over GF(2), so the state initialized by
    ``UCCSDFactory`` matches the same code used for the Hamiltonian and
    excitation generators.
    """

    n_modes = int(n_spin_orbitals)
    if n_modes <= 0:
        raise ValueError("n_spin_orbitals must be positive")
    occupied = tuple(sorted({int(index) for index in occupied_spin_orbitals}))
    if any(index < 0 or index >= n_modes for index in occupied):
        raise ValueError("occupied_spin_orbitals contains an out-of-range index")
    mapper = normalize_mapper_type(mapper_type)
    if mapper == "jw":
        return occupied

    try:
        from openfermion import bravyi_kitaev_code, parity_code
    except ImportError as exc:
        raise ImportError("BK/parity reference-state encoding requires openfermion") from exc

    occupation = np.zeros(n_modes, dtype=np.int8)
    occupation[list(occupied)] = 1
    code = bravyi_kitaev_code(n_modes) if mapper == "bk" else parity_code(n_modes)
    encoded = np.asarray(code.encoder.dot(occupation), dtype=np.int64).reshape(-1) % 2
    if encoded.size != n_modes:
        raise RuntimeError("Binary code returned an unexpected reference-state width")
    return tuple(int(index) for index in np.flatnonzero(encoded))


__all__ = [
    "MapperType",
    "encoded_occupied_qubits",
    "map_fermion_operator",
    "normalize_mapper_type",
]
