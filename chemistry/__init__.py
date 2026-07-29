"""Molecular electronic-structure and active-space helpers."""

from .active_space import (
    ActiveSpaceConfig,
    ActiveSpaceReport,
    estimate_frozen_core_orbitals,
    select_active_space,
)
from .molecule import MolecularDataEngine
from .mapping import encoded_occupied_qubits, map_fermion_operator, normalize_mapper_type
from .uccsd import (
    CanonicalBasisAudit,
    SingletExcitation,
    canonical_parameter_count,
    canonical_singlet_descriptors,
    canonical_singlet_operator,
)
from .uccsd_amplitudes import pack_closed_shell_ccsd

__all__ = [
    "ActiveSpaceConfig",
    "ActiveSpaceReport",
    "CanonicalBasisAudit",
    "MolecularDataEngine",
    "encoded_occupied_qubits",
    "map_fermion_operator",
    "normalize_mapper_type",
    "SingletExcitation",
    "canonical_parameter_count",
    "canonical_singlet_descriptors",
    "canonical_singlet_operator",
    "estimate_frozen_core_orbitals",
    "pack_closed_shell_ccsd",
    "select_active_space",
]
