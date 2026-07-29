# Changelog

## 1.4.3

- Added end-to-end Bravyi-Kitaev and parity mapping support for canonical
  singlet-UCCSD excitation generators.
- Encoded the Hartree-Fock computational-basis reference using the same BK or
  parity code as the molecular Hamiltonian and ansatz.
- Centralized mapper aliases, fermion-to-qubit transformations, and reference
  occupation encoding in ``chemistry.mapping``.

## 1.4.2

- Corrected QuantumCoordinateDescent extrema selection and function-evaluation accounting.
- Ensured SPSA and VQESolver return objective values evaluated at the reported parameters.
- Invalidated prepared Hamiltonians when mutable input content changes.
- Fixed the Nelder-Mead outside-contraction branch and strengthened simplex convergence checks.
- Reset optimizer trajectories on each run and validate optimizer hyperparameters.
- Updated the cqlib2 API check for native Pauli rotations.
- Rejected `min_virtual_orbitals=0` instead of silently changing the requested value.
- Added regression tests for optimizer branches, result contracts, cache invalidation, and API checks.

## 1.4.1

- Compiled Tianyan-bound circuits to the native QCIS basis before submission.
- Rejected non-native QCIS instructions locally with line-level diagnostics.
- Used native basis-change sequences for X/Y Pauli measurements.
- Normalized API keys without exposing them and rejected blank keys.
- Added the canonical `cqlib_vqe` namespace while retaining 1.x legacy imports.
- Added native-QCIS compiler test doubles and import-compatibility tests.
- Added public release metadata, license, security guidance, and clean source artifacts.
- Removed generated caches, logs, internal delivery notes, and hard-coded backend examples.

## 1.4.0

- Added optional fused cqlib `Statevector.apply_pauli_rotation` execution.
- Precompiled Pauli supports, Trotter schedules, and native PauliString objects.
- Preferred the native path in `execution_mode="auto"` when available.
- Reused prepared Hamiltonians and one full canonical factory during adaptive screening.
- Initialized newly selected generators from the best finite-difference probe.
- Preserved circuit, parameter-binding, Tianyan, and decomposed fallback paths.
