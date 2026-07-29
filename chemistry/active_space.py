"""Deterministic full, manual, and automatic molecular active spaces.

The active-space Hamiltonian itself is built by OpenFermion's
``MolecularData.get_molecular_hamiltonian``.  That routine folds frozen occupied
orbitals into the scalar and one-electron parts of the reduced Hamiltonian.
This module only selects and reports the orbital partition; it does not pretend
to perform CASSCF orbital optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np


_ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
    "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
}


def _previous_noble_gas_electrons(atomic_number: int) -> int:
    if atomic_number <= 2:
        return 0
    if atomic_number <= 10:
        return 2
    if atomic_number <= 18:
        return 10
    if atomic_number <= 36:
        return 18
    if atomic_number <= 54:
        return 36
    if atomic_number <= 86:
        return 54
    return 86


def estimate_frozen_core_orbitals(symbols: Iterable[str]) -> int:
    """Estimate the conventional frozen-core spatial-orbital count."""

    frozen_electrons = 0
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip()
        if symbol not in _ATOMIC_NUMBERS:
            raise ValueError(
                f"Unknown element {symbol!r} for automatic frozen-core selection; "
                "use a manual active space."
            )
        frozen_electrons += _previous_noble_gas_electrons(_ATOMIC_NUMBERS[symbol])
    return frozen_electrons // 2


@dataclass(frozen=True)
class ActiveSpaceConfig:
    """Configuration for full, manual, or deterministic automatic selection."""

    mode: Literal["full", "auto", "manual"] = "full"
    freeze_core: bool = True
    max_active_qubits: int | None = None
    max_active_orbitals: int | None = None
    active_electrons: int | None = None
    active_orbitals: int | None = None
    min_virtual_orbitals: int = 1
    virtual_energy_window: float | None = None
    allow_additional_frozen_occupied: bool = True

    @classmethod
    def full(cls) -> "ActiveSpaceConfig":
        return cls(mode="full", freeze_core=False)

    @classmethod
    def automatic(
        cls,
        *,
        max_active_qubits: int | None = None,
        max_active_orbitals: int | None = None,
        active_electrons: int | None = None,
        active_orbitals: int | None = None,
        freeze_core: bool = True,
        min_virtual_orbitals: int = 1,
        virtual_energy_window: float | None = None,
        allow_additional_frozen_occupied: bool = True,
    ) -> "ActiveSpaceConfig":
        return cls(
            mode="auto",
            freeze_core=freeze_core,
            max_active_qubits=max_active_qubits,
            max_active_orbitals=max_active_orbitals,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            min_virtual_orbitals=min_virtual_orbitals,
            virtual_energy_window=virtual_energy_window,
            allow_additional_frozen_occupied=allow_additional_frozen_occupied,
        )

    @classmethod
    def manual(cls) -> "ActiveSpaceConfig":
        return cls(mode="manual")

    def validate(self) -> None:
        if self.mode not in {"full", "auto", "manual"}:
            raise ValueError(f"Unknown active-space mode {self.mode!r}")
        if self.max_active_qubits is not None and (
            self.max_active_qubits <= 0 or self.max_active_qubits % 2
        ):
            raise ValueError("max_active_qubits must be a positive even integer")
        for name, value in (
            ("max_active_orbitals", self.max_active_orbitals),
            ("active_orbitals", self.active_orbitals),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_virtual_orbitals < 1:
            raise ValueError("min_virtual_orbitals must be at least 1")
        if self.active_electrons is not None and (
            self.active_electrons <= 0 or self.active_electrons % 2
        ):
            raise ValueError("active_electrons must be a positive even integer")
        if self.virtual_energy_window is not None and self.virtual_energy_window <= 0:
            raise ValueError("virtual_energy_window must be positive")
        budget_count = sum(
            value is not None
            for value in (
                self.max_active_qubits,
                self.max_active_orbitals,
                self.active_orbitals,
            )
        )
        if budget_count > 1:
            raise ValueError(
                "Specify only one of max_active_qubits, max_active_orbitals, "
                "or active_orbitals"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveSpaceReport:
    mode: str
    original_spatial_orbitals: int
    original_electrons: int
    occupied_spatial_orbitals: int
    frozen_occupied_indices: tuple[int, ...]
    active_indices: tuple[int, ...]
    excluded_virtual_indices: tuple[int, ...]
    active_electrons: int
    active_spatial_orbitals: int
    pre_taper_qubits: int
    estimated_core_orbitals: int
    additional_frozen_occupied: int
    selection_reason: str
    orbital_energies: tuple[float, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _manual_report(
    occupied_indices: Sequence[int] | None,
    active_indices: Sequence[int] | None,
    *,
    n_orbitals: int,
    n_electrons: int,
    orbital_energies: np.ndarray,
) -> ActiveSpaceReport:
    frozen = tuple(sorted(int(index) for index in (occupied_indices or ())))
    active = tuple(sorted(int(index) for index in (active_indices or ())))
    if len(set(frozen)) != len(frozen) or len(set(active)) != len(active):
        raise ValueError("Manual active-space indices must be unique")
    if set(frozen) & set(active):
        raise ValueError("occupied_indices and active_indices must be disjoint")
    if any(index < 0 or index >= n_orbitals for index in (*frozen, *active)):
        raise ValueError("Manual active-space index is outside the orbital range")
    n_occ = n_electrons // 2
    if any(index >= n_occ for index in frozen):
        raise ValueError("occupied_indices may contain only occupied spatial orbitals")
    if not active:
        active = tuple(index for index in range(n_orbitals) if index not in frozen)
    missing_occupied = sorted(set(range(n_occ)) - set(frozen) - set(active))
    if missing_occupied:
        raise ValueError(
            "Every occupied orbital must be either frozen or active; missing "
            f"occupied indices {missing_occupied}"
        )
    active_electrons = n_electrons - 2 * len(frozen)
    if active_electrons <= 0 or active_electrons > 2 * len(active):
        raise ValueError(
            f"Inconsistent manual active space: {active_electrons} electrons in "
            f"{len(active)} spatial orbitals"
        )
    excluded_virtual = tuple(
        index for index in range(n_occ, n_orbitals) if index not in active
    )
    warnings: tuple[str, ...] = ()
    if active and active != tuple(range(active[0], active[-1] + 1)):
        warnings = ("Manual active orbital indices are non-contiguous.",)
    return ActiveSpaceReport(
        mode="manual",
        original_spatial_orbitals=n_orbitals,
        original_electrons=n_electrons,
        occupied_spatial_orbitals=n_occ,
        frozen_occupied_indices=frozen,
        active_indices=active,
        excluded_virtual_indices=excluded_virtual,
        active_electrons=active_electrons,
        active_spatial_orbitals=len(active),
        pre_taper_qubits=2 * len(active),
        estimated_core_orbitals=len(frozen),
        additional_frozen_occupied=0,
        selection_reason="User-provided manual active space.",
        orbital_energies=tuple(float(value) for value in orbital_energies),
        warnings=warnings,
    )


def select_active_space(
    *,
    config: ActiveSpaceConfig,
    geometry: Sequence[tuple[str, Sequence[float]]],
    n_orbitals: int,
    n_electrons: int,
    orbital_energies: Sequence[float],
    occupied_indices: Sequence[int] | None = None,
    active_indices: Sequence[int] | None = None,
) -> ActiveSpaceReport:
    """Select a deterministic active space and return a complete report."""

    config.validate()
    energies = np.asarray(orbital_energies, dtype=float).reshape(-1)
    if energies.size != n_orbitals:
        raise ValueError(
            f"Expected {n_orbitals} orbital energies, received {energies.size}"
        )
    if n_electrons <= 0 or n_electrons % 2:
        raise ValueError(
            "This active-space implementation currently requires a positive, "
            "even closed-shell electron count"
        )
    n_occ = n_electrons // 2
    if n_occ > n_orbitals:
        raise ValueError("Electron count exceeds available orbitals")

    if config.mode == "manual" or occupied_indices is not None or active_indices is not None:
        return _manual_report(
            occupied_indices,
            active_indices,
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            orbital_energies=energies,
        )

    if config.mode == "full":
        active = tuple(range(n_orbitals))
        return ActiveSpaceReport(
            mode="full",
            original_spatial_orbitals=n_orbitals,
            original_electrons=n_electrons,
            occupied_spatial_orbitals=n_occ,
            frozen_occupied_indices=(),
            active_indices=active,
            excluded_virtual_indices=(),
            active_electrons=n_electrons,
            active_spatial_orbitals=n_orbitals,
            pre_taper_qubits=2 * n_orbitals,
            estimated_core_orbitals=0,
            additional_frozen_occupied=0,
            selection_reason="Full molecular orbital space retained.",
            orbital_energies=tuple(float(value) for value in energies),
        )

    symbols = [atom for atom, _ in geometry]
    estimated_core = estimate_frozen_core_orbitals(symbols) if config.freeze_core else 0
    estimated_core = min(estimated_core, max(0, n_occ - 1))

    if config.active_electrons is not None:
        if config.active_electrons > n_electrons:
            raise ValueError("active_electrons cannot exceed molecular electrons")
        requested_frozen = (n_electrons - config.active_electrons) // 2
        frozen_count = max(estimated_core, requested_frozen)
    else:
        frozen_count = estimated_core

    if config.active_orbitals is not None:
        capacity = config.active_orbitals
    elif config.max_active_orbitals is not None:
        capacity = config.max_active_orbitals
    elif config.max_active_qubits is not None:
        capacity = config.max_active_qubits // 2
    else:
        capacity = n_orbitals - frozen_count
    capacity = min(int(capacity), n_orbitals)
    if capacity <= 0:
        raise ValueError("Active-space orbital budget must be positive")

    min_virtual = int(config.min_virtual_orbitals)
    active_occupied_count = n_occ - frozen_count
    additional_frozen = 0
    warnings: list[str] = []
    if active_occupied_count + min_virtual > capacity:
        if not config.allow_additional_frozen_occupied:
            raise ValueError(
                "Active-space budget cannot retain the non-core occupied orbitals "
                f"and {min_virtual} virtual orbital(s)"
            )
        additional_frozen = active_occupied_count + min_virtual - capacity
        frozen_count += additional_frozen
        active_occupied_count = n_occ - frozen_count
        warnings.append(
            "The qubit budget froze occupied orbitals beyond the conventional core."
        )
    if active_occupied_count <= 0:
        raise ValueError("Automatic selection removed all occupied active orbitals")

    occupied_by_energy = sorted(
        range(n_occ), key=lambda index: (float(energies[index]), index)
    )
    frozen_set = set(occupied_by_energy[:frozen_count])
    active_occupied = sorted(index for index in range(n_occ) if index not in frozen_set)

    virtual_candidates = sorted(
        range(n_occ, n_orbitals), key=lambda index: (float(energies[index]), index)
    )
    if config.virtual_energy_window is not None and virtual_candidates:
        lumo = float(energies[virtual_candidates[0]])
        limit = lumo + float(config.virtual_energy_window)
        within_window = [
            index for index in virtual_candidates if float(energies[index]) <= limit
        ]
        virtual_candidates = (
            within_window
            if len(within_window) >= min_virtual
            else virtual_candidates[:min_virtual]
        )

    virtual_capacity = capacity - len(active_occupied)
    selected_virtual = sorted(virtual_candidates[:virtual_capacity])
    if len(selected_virtual) < min_virtual:
        raise ValueError(
            f"Only {len(selected_virtual)} virtual orbital(s) fit; {min_virtual} required"
        )

    frozen = tuple(sorted(frozen_set))
    active = tuple(active_occupied + selected_virtual)
    excluded_virtual = tuple(
        index for index in range(n_occ, n_orbitals) if index not in selected_virtual
    )
    return ActiveSpaceReport(
        mode="auto",
        original_spatial_orbitals=n_orbitals,
        original_electrons=n_electrons,
        occupied_spatial_orbitals=n_occ,
        frozen_occupied_indices=frozen,
        active_indices=active,
        excluded_virtual_indices=excluded_virtual,
        active_electrons=n_electrons - 2 * frozen_count,
        active_spatial_orbitals=len(active),
        pre_taper_qubits=2 * len(active),
        estimated_core_orbitals=estimated_core,
        additional_frozen_occupied=additional_frozen,
        selection_reason=(
            "Frozen-core and orbital-energy selection: highest occupied and lowest "
            "virtual orbitals retained within the configured budget."
        ),
        orbital_energies=tuple(float(value) for value in energies),
        warnings=tuple(warnings),
    )


__all__ = [
    "ActiveSpaceConfig",
    "ActiveSpaceReport",
    "estimate_frozen_core_orbitals",
    "select_active_space",
]
