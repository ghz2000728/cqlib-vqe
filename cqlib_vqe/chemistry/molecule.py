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

"""Correct closed-shell chemistry preprocessing for cqlib VQE.

This implementation keeps one consistent active-space problem across every
stage:

* OpenFermion builds the reduced Hamiltonian with frozen occupied orbitals
  folded into the scalar and one-electron terms;
* PySCF CCSD is rerun with exactly the inactive orbitals frozen;
* the resulting spin-orbital amplitudes are packed with OpenFermion's canonical
  singlet-UCCSD packer;
* selected quantum generators are identified only by canonical ``packed_index``.

Pauli strings keep the project convention that character ``i`` acts on qubit
``i``.  Conversion to cqlib's display order remains isolated in
``cqlib_vqe.vqe.hamiltonian``.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np

from .active_space import ActiveSpaceConfig, ActiveSpaceReport, select_active_space
from .mapping import encoded_occupied_qubits, map_fermion_operator, normalize_mapper_type
from .uccsd_amplitudes import pack_closed_shell_ccsd


class MolecularDataEngine:
    """Prepare a closed-shell active-space Hamiltonian and canonical UCCSD data."""

    def __init__(
        self,
        geometry: Sequence[tuple[str, Sequence[float]]],
        basis: str = "sto-3g",
        multiplicity: int = 1,
        charge: int = 0,
        mapper_type: str = "jw",
        occupied_indices: Sequence[int] | None = None,
        active_indices: Sequence[int] | None = None,
        *,
        excitation_threshold: float = 1e-3,
        active_space: ActiveSpaceConfig | str | None = None,
    ) -> None:
        mapper = normalize_mapper_type(mapper_type)
        if int(multiplicity) != 1:
            raise NotImplementedError(
                "The corrected canonical singlet-UCCSD path currently supports "
                "closed-shell singlets only"
            )
        if excitation_threshold < 0:
            raise ValueError("excitation_threshold must be non-negative")

        if active_space is None:
            config = (
                ActiveSpaceConfig.manual()
                if occupied_indices is not None or active_indices is not None
                else ActiveSpaceConfig.full()
            )
        elif isinstance(active_space, ActiveSpaceConfig):
            config = active_space
        elif isinstance(active_space, str):
            mode = active_space.lower()
            if mode == "full":
                config = ActiveSpaceConfig.full()
            elif mode == "auto":
                config = ActiveSpaceConfig.automatic()
            elif mode == "manual":
                config = ActiveSpaceConfig.manual()
            else:
                raise ValueError(f"Unknown active_space mode {active_space!r}")
        else:
            raise TypeError("active_space must be ActiveSpaceConfig, a mode string, or None")
        if (occupied_indices is not None or active_indices is not None) and config.mode not in {
            "manual",
            "full",
        }:
            raise ValueError(
                "Explicit occupied_indices/active_indices cannot be mixed with an "
                "automatic active-space config"
            )

        self.geometry = [
            (str(atom), tuple(float(value) for value in coordinates))
            for atom, coordinates in geometry
        ]
        self.basis = str(basis)
        self.multiplicity = int(multiplicity)
        self.charge = int(charge)
        self.mapper_type = mapper
        self.excitation_threshold = float(excitation_threshold)
        self.active_space_config = config
        self.occupied_indices = (
            None if occupied_indices is None else [int(value) for value in occupied_indices]
        )
        self.active_indices = (
            None if active_indices is None else [int(value) for value in active_indices]
        )

        self.molecule: Any | None = None
        self.active_space_report: ActiveSpaceReport | None = None
        self.n_qubits = 0
        self.n_spin_orbitals = 0
        self.n_electrons = 0
        self.nuclear_repulsion = 0.0
        self.hf_energy = 0.0
        self.ccsd_energy: float | None = None
        self.fci_energy: float | None = None
        self.reference_energies: dict[str, float] = {}
        self.initial_bitstring: list[int] = []
        self.hamiltonian_data: list[tuple[str, float]] = []

        self.spin_singles: list[tuple[int, int]] = []
        self.spin_doubles: list[tuple[int, int, int, int]] = []
        self.spatial_singles: list[dict[str, Any]] = []
        self.spatial_doubles: list[dict[str, Any]] = []
        # Names retained for the 1.2.x factory interface.
        self.sp_singles = self.spatial_singles
        self.sp_doubles = self.spatial_doubles

        self.ccsd_descriptors: list[Any] = []
        self.ccsd_parameters = np.asarray([], dtype=np.float64)
        self.ccsd_packing_report = None
        self.ccsd_projection_report = None
        self.ccsd_full_parameter_count = 0
        self._ccsd_single_amplitudes: np.ndarray | None = None
        self._ccsd_double_amplitudes: np.ndarray | None = None

    def run(
        self,
        *,
        run_fci: bool = True,
        audit_ccsd_basis: bool = True,
    ) -> "MolecularDataEngine":
        try:
            from openfermion import MolecularData
            from openfermionpyscf import run_pyscf
        except ImportError as exc:
            raise ImportError(
                "MolecularDataEngine requires openfermion and openfermionpyscf"
            ) from exc

        print(f"[Chemistry] Computing molecular integrals: {self.geometry}")
        base = MolecularData(self.geometry, self.basis, self.multiplicity, self.charge)
        # Correlated methods are deliberately delayed until the active space is known.
        self.molecule = run_pyscf(
            base,
            run_scf=True,
            run_mp2=False,
            run_cisd=False,
            run_ccsd=False,
            run_fci=False,
        )
        self.nuclear_repulsion = float(self.molecule.nuclear_repulsion)
        self.hf_energy = float(self.molecule.hf_energy)

        self._resolve_active_space()
        self._extract_hamiltonian_data()
        self.initial_occupied_qubits = encoded_occupied_qubits(
            n_spin_orbitals=self.n_qubits,
            occupied_spin_orbitals=range(self.n_electrons),
            mapper_type=self.mapper_type,
        )
        self.initial_bitstring = [
            int(qubit in self.initial_occupied_qubits) for qubit in range(self.n_qubits)
        ]
        self._compute_active_space_ccsd()
        self._extract_canonical_excitations(audit_ccsd_basis=audit_ccsd_basis)

        if run_fci:
            self.fci_energy = self._compute_active_space_fci()
        self.reference_energies = {
            "hf": self.hf_energy,
            "ccsd": float(self.ccsd_energy),
            **({"fci": float(self.fci_energy)} if self.fci_energy is not None else {}),
        }
        if self.fci_energy is not None:
            print(f"[Chemistry] Active-space FCI: {self.fci_energy:.12f} Ha")
        return self

    # ------------------------------------------------------------------
    # Active space and Hamiltonian
    # ------------------------------------------------------------------
    def _resolve_active_space(self) -> None:
        if self.molecule is None:
            raise RuntimeError("SCF data are unavailable")
        report = select_active_space(
            config=self.active_space_config,
            geometry=self.geometry,
            n_orbitals=int(self.molecule.n_orbitals),
            n_electrons=int(self.molecule.n_electrons),
            orbital_energies=self.molecule.orbital_energies,
            occupied_indices=self.occupied_indices,
            active_indices=self.active_indices,
        )
        self.active_space_report = report
        self.occupied_indices = list(report.frozen_occupied_indices)
        self.active_indices = list(report.active_indices)
        self.n_electrons = int(report.active_electrons)
        self.n_spin_orbitals = int(report.pre_taper_qubits)
        self.n_qubits = self.n_spin_orbitals
        for message in report.warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        print(
            "[Chemistry] Active space: "
            f"mode={report.mode}, frozen={list(report.frozen_occupied_indices)}, "
            f"active={list(report.active_indices)}, electrons={self.n_electrons}, "
            f"qubits={self.n_qubits}"
        )

    def _active_interaction_hamiltonian(self):
        if self.molecule is None:
            raise RuntimeError("run() must be called first")
        return self.molecule.get_molecular_hamiltonian(
            occupied_indices=self.occupied_indices,
            active_indices=self.active_indices,
        )

    def _fermion_to_qubit_op(self, interaction):
        return map_fermion_operator(
            interaction,
            mapper_type=self.mapper_type,
            n_spin_orbitals=self.n_spin_orbitals,
        )

    def _extract_hamiltonian_data(self) -> None:
        qubit_hamiltonian = self._fermion_to_qubit_op(
            self._active_interaction_hamiltonian()
        )
        data: list[tuple[str, float]] = []
        for term, coefficient in qubit_hamiltonian.terms.items():
            coefficient = complex(coefficient)
            if abs(coefficient.imag) > 1e-10:
                raise ValueError(
                    f"Mapped Hamiltonian term {term!r} has imaginary coefficient "
                    f"{coefficient!r}"
                )
            chars = ["I"] * self.n_qubits
            for qubit, char in term:
                chars[int(qubit)] = str(char)
            data.append(("".join(chars), float(coefficient.real)))
        self.hamiltonian_data = sorted(data, key=lambda item: item[0])

    # ------------------------------------------------------------------
    # Active-space correlated references and canonical amplitudes
    # ------------------------------------------------------------------
    def _inactive_orbital_indices(self) -> list[int]:
        if self.molecule is None:
            raise RuntimeError("run() must be called first")
        active = set(int(index) for index in self.active_indices or ())
        return [
            index
            for index in range(int(self.molecule.n_orbitals))
            if index not in active
        ]

    def _compute_active_space_ccsd(self) -> None:
        try:
            from pyscf import cc
            from pyscf.cc.addons import spatial2spin
        except ImportError as exc:
            raise ImportError("Active-space CCSD requires pyscf") from exc
        if self.molecule is None:
            raise RuntimeError("run() must be called first")
        mean_field = self.molecule._pyscf_data.get("scf")
        if mean_field is None:
            raise RuntimeError("OpenFermion-PySCF did not preserve the SCF object")

        frozen = self._inactive_orbital_indices()
        calculation = cc.CCSD(mean_field, frozen=frozen or None)
        calculation.verbose = 0
        calculation.run()
        if not bool(getattr(calculation, "converged", True)):
            raise RuntimeError("Active-space CCSD did not converge")
        self.ccsd_energy = float(calculation.e_tot)

        t1 = np.asarray(spatial2spin(calculation.t1), dtype=np.float64)
        t2 = np.asarray(spatial2spin(calculation.t2), dtype=np.float64)
        n_occ_spin, n_virt_spin = t1.shape
        n_modes = n_occ_spin + n_virt_spin
        if n_modes != self.n_spin_orbitals:
            raise RuntimeError(
                "Active-space CCSD dimension mismatch: "
                f"{n_modes} modes != {self.n_spin_orbitals}"
            )
        singles = np.zeros((n_modes, n_modes), dtype=np.float64)
        doubles = np.zeros((n_modes,) * 4, dtype=np.float64)
        singles[n_occ_spin:, :n_occ_spin] = t1.T
        # OpenFermion's spin-orbital convention stores one half of the
        # antisymmetrized PySCF T2 tensor at [virtual, occupied, virtual, occupied].
        doubles[n_occ_spin:, :n_occ_spin, n_occ_spin:, :n_occ_spin] = (
            0.5 * t2.transpose(2, 0, 3, 1)
        )
        self._ccsd_single_amplitudes = singles
        self._ccsd_double_amplitudes = doubles

    def _extract_canonical_excitations(self, *, audit_ccsd_basis: bool) -> None:
        if self._ccsd_single_amplitudes is None or self._ccsd_double_amplitudes is None:
            raise RuntimeError("CCSD amplitudes are unavailable")
        descriptors, parameters, report = pack_closed_shell_ccsd(
            self._ccsd_single_amplitudes,
            self._ccsd_double_amplitudes,
            self.n_spin_orbitals,
            self.n_electrons,
            audit=audit_ccsd_basis,
        )
        self.ccsd_descriptors = list(descriptors)
        self.ccsd_parameters = np.asarray(parameters, dtype=np.float64)
        self.ccsd_packing_report = report
        self.ccsd_projection_report = report
        self.ccsd_full_parameter_count = int(report.parameter_count)

        selected = [
            descriptor.packed_index
            for descriptor, amplitude in zip(descriptors, parameters, strict=True)
            if abs(float(amplitude)) > self.excitation_threshold
        ]
        self.spatial_singles, self.spatial_doubles = self.ccsd_items(selected)
        self.sp_singles = self.spatial_singles
        self.sp_doubles = self.spatial_doubles

        self.spin_singles = [
            (int(virtual), int(occupied))
            for virtual, occupied in np.argwhere(
                np.abs(self._ccsd_single_amplitudes) > self.excitation_threshold
            )
        ]
        self.spin_doubles = [
            (int(va), int(vb), int(oa), int(ob))
            for va, oa, vb, ob in np.argwhere(
                np.abs(self._ccsd_double_amplitudes) > self.excitation_threshold
            )
        ]
        print(
            "[Chemistry] Canonical singlet-UCCSD pool: "
            f"selected={len(selected)}/{self.ccsd_full_parameter_count}, "
            f"threshold={self.excitation_threshold:.1e}"
        )

    def ccsd_amplitude_map(self) -> dict[int, float]:
        return {
            int(descriptor.packed_index): float(amplitude)
            for descriptor, amplitude in zip(
                self.ccsd_descriptors, self.ccsd_parameters, strict=True
            )
        }

    def ccsd_items(
        self,
        packed_indices: Sequence[int] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.ccsd_descriptors:
            return [], []
        known = {int(item.packed_index) for item in self.ccsd_descriptors}
        allowed = known if packed_indices is None else {int(value) for value in packed_indices}
        unknown = sorted(allowed - known)
        if unknown:
            raise ValueError(f"Unknown canonical packed indices: {unknown}")

        singles: list[dict[str, Any]] = []
        doubles: list[dict[str, Any]] = []
        for descriptor, amplitude in zip(
            self.ccsd_descriptors, self.ccsd_parameters, strict=True
        ):
            packed_index = int(descriptor.packed_index)
            if packed_index not in allowed:
                continue
            value = float(amplitude)
            if descriptor.kind == "single":
                occupied, virtual = descriptor.pair_a
                singles.append(
                    {
                        "indices": [occupied, virtual],
                        "amplitude": value,
                        "packed_index": packed_index,
                    }
                )
            else:
                if descriptor.pair_b is None:
                    raise RuntimeError("Canonical double descriptor has no pair_b")
                occupied_a, virtual_a = descriptor.pair_a
                occupied_b, virtual_b = descriptor.pair_b
                doubles.append(
                    {
                        "occ": [occupied_a, occupied_b],
                        "virt": [virtual_a, virtual_b],
                        "amplitude": value,
                        "packed_index": packed_index,
                    }
                )
        singles.sort(key=lambda item: int(item["packed_index"]))
        doubles.sort(key=lambda item: int(item["packed_index"]))
        return singles, doubles

    def candidate_packed_indices(self, minimum_amplitude: float = 0.0) -> list[int]:
        if minimum_amplitude < 0:
            raise ValueError("minimum_amplitude must be non-negative")
        return [
            int(descriptor.packed_index)
            for descriptor, amplitude in zip(
                self.ccsd_descriptors, self.ccsd_parameters, strict=True
            )
            if abs(float(amplitude)) >= minimum_amplitude
        ]

    def _compute_active_space_fci(self) -> float:
        try:
            from pyscf import mcscf
        except ImportError as exc:
            raise ImportError("Active-space FCI requires pyscf") from exc
        if self.molecule is None:
            raise RuntimeError("run() must be called first")
        mean_field = self.molecule._pyscf_data.get("scf")
        if mean_field is None:
            raise RuntimeError("OpenFermion-PySCF did not preserve the SCF object")
        active_1based = [int(index) + 1 for index in self.active_indices or ()]
        casci = mcscf.CASCI(mean_field, len(active_1based), self.n_electrons)
        casci.verbose = 0
        mo_coeff = mean_field.mo_coeff
        full = list(range(1, int(self.molecule.n_orbitals) + 1))
        if active_1based != full:
            mo_coeff = mcscf.sort_mo(casci, mo_coeff, active_1based)
        result = casci.kernel(mo_coeff)
        return float(result[0])


__all__ = ["MolecularDataEngine"]
