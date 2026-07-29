"""Pack RHF-CCSD tensors in the canonical singlet-UCCSD basis."""

from __future__ import annotations

import numpy as np

from .uccsd import (
    CanonicalBasisAudit,
    ProjectionReport,
    SingletExcitation,
    audit_fermion_operator,
    build_singlet_basis,
    canonical_singlet_descriptors,
    canonical_singlet_operator,
    project_fermion_operator,
)


def pack_closed_shell_ccsd(
    single_amplitudes: np.ndarray,
    double_amplitudes: np.ndarray,
    n_qubits: int,
    n_electrons: int,
    *,
    atol: float = 1e-12,
    audit: bool = True,
) -> tuple[list[SingletExcitation], np.ndarray, CanonicalBasisAudit]:
    try:
        from openfermion import normal_ordered, uccsd_singlet_generator
        from openfermion.circuits import uccsd_singlet_get_packed_amplitudes
    except ImportError as exc:
        raise ImportError("canonical CCSD packing requires openfermion") from exc

    singles = np.asarray(single_amplitudes, dtype=np.float64)
    doubles = np.asarray(double_amplitudes, dtype=np.float64)
    if singles.shape != (n_qubits, n_qubits):
        raise ValueError(
            f"CCSD singles shape {singles.shape} != {(n_qubits, n_qubits)}"
        )
    if doubles.shape != (n_qubits,) * 4:
        raise ValueError(
            f"CCSD doubles shape {doubles.shape} != {(n_qubits,) * 4}"
        )
    if not np.all(np.isfinite(singles)) or not np.all(np.isfinite(doubles)):
        raise ValueError("CCSD amplitudes contain NaN or infinity")

    descriptors = canonical_singlet_descriptors(n_qubits, n_electrons)
    packed = np.asarray(
        uccsd_singlet_get_packed_amplitudes(
            singles, doubles, n_qubits, n_electrons
        ),
        dtype=np.float64,
    )
    if packed.shape != (len(descriptors),):
        raise RuntimeError(
            f"Packed amplitude shape {packed.shape} != {(len(descriptors),)}"
        )
    if not audit:
        return descriptors, packed, CanonicalBasisAudit(
            relative_residual=0.0,
            absolute_residual=0.0,
            target_norm=float(np.linalg.norm(packed)),
            rank=len(descriptors),
            parameter_count=len(descriptors),
            term_count=0,
            audited=False,
        )

    _, basis = build_singlet_basis(n_qubits, n_electrons)
    target = normal_ordered(
        uccsd_singlet_generator(
            packed.tolist(), n_qubits, n_electrons, anti_hermitian=True
        )
    )
    audited, report = audit_fermion_operator(target, basis, atol=atol)
    if not np.allclose(audited, packed, atol=atol, rtol=0.0):
        error = float(np.max(np.abs(audited - packed)))
        raise ValueError(f"Canonical amplitude audit failed: max error={error:.3e}")
    if report.rank != report.parameter_count:
        raise ValueError(
            f"Canonical singlet basis rank deficient: {report.rank}/{report.parameter_count}"
        )
    return descriptors, packed, report


project_closed_shell_ccsd = pack_closed_shell_ccsd

__all__ = [
    "CanonicalBasisAudit",
    "ProjectionReport",
    "SingletExcitation",
    "pack_closed_shell_ccsd",
    "project_closed_shell_ccsd",
]
