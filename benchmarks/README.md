# cqlib-vqe benchmark

This benchmark measures the fused cqlib2 PauliRotation path and keeps the
previous execution paths as internal controls.

## Engine cases

- `native_pauli_objective`: recommended path. The compiled UCC
  plan calls `Statevector.apply_pauli_rotation` once per Pauli exponential.
- `decomposed_pauli_objective`: same statevector backend and same UCC plan, but
  every Pauli exponential is forced through the legacy H/RX/CX/RZ expansion.
- `circuit_bind_objective`: symbolic circuit constructed once, followed by
  `Circuit.assign_parameters` and circuit execution.
- `circuit_jit_objective`: numeric circuit rebuilt for every parameter vector.
- `assign_parameters_only`: parameter binding without statevector evolution or
  Hamiltonian expectation.

All four objective paths must agree within `1e-9 Ha`.

## Workflow cases

- `full_uccsd`
- `static_selected`
- `adaptive_r1`
- `adaptive_r2` (optional internal ablation)

The workflow defaults to `execution_mode=auto`, which selects the fused
native Pauli path when the patched cqlib2 binding is installed. Adaptive runs
also default to `candidate_initialization=best_probe`.

FCI is hidden from the solver and is used only after the run to score chemical
accuracy.

## Recommended commands

Smoke test:

```bash
MOLECULES=h2 PRESET=smoke ./scripts/run_cqlib_benchmark.sh
```

Official engine comparison:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
RAYON_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PRESET=engine \
ENGINE_COUNTS=100 \
ENGINE_REPEATS=5 \
ENGINE_WARMUP=10 \
./scripts/run_cqlib_benchmark.sh
```

Workflow:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
RAYON_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PRESET=workflow \
WORKFLOW_REPEATS=3 \
VARIANTS=full_uccsd,static_selected,adaptive_r1 \
MAXITER=80 \
./scripts/run_cqlib_benchmark.sh
```

The output is written atomically after each molecule, so completed results
survive interruption.
