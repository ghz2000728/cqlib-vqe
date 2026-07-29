# cqlib-vqe

[简体中文](README.zh-CN.md)

`cqlib-vqe` is a VQE package for molecular electronic-structure calculations on
cqlib 2.x.  It provides molecular-Hamiltonian preparation, canonical
singlet-UCCSD ansätze, Jordan-Wigner (JW), Bravyi-Kitaev (BK), and parity
encodings, adaptive operator-pool selection, local statevector execution, and
Tianyan cloud/hardware measurement support.

The primary public import is `cqlib_vqe`.

## What the package provides

- Molecular preprocessing through PySCF and OpenFermion, including a
  reproducible active-space report and Pauli Hamiltonian.
- Canonical singlet-UCCSD generators with a stable packed-parameter ordering.
- Consistent JW, BK, and parity treatment of the Hamiltonian, excitation
  generators, and Hartree-Fock reference state.
- `DirectStatevectorEstimator` for local high-throughput VQE objectives and
  `NativeStatevectorEstimator` for circuit/statevector expectation values.
- Complete-pool adaptive selected-UCCSD: a zero CCSD initial amplitude does
  not remove an operator from the candidate pool.
- QWC-grouped Tianyan measurement plans and strict validation of submitted-task
  and result-qubit metadata.

## Installation

Clone the repository and create an isolated Python environment. Python 3.10 or
newer is required.

```bash
git clone https://github.com/cq-lib/cqlib-vqe.git
cd cqlib-vqe

python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the variant required by the workflow:

```bash
# Local VQE from a precompiled ansatz.
python -m pip install -e .

# Molecular Hamiltonian and UCCSD preprocessing.
python -m pip install -e ".[chemistry]"

# Tianyan execution in addition to molecular preprocessing.
python -m pip install -e ".[chemistry,tianyan]"

# Development and test dependencies.
python -m pip install -e ".[chemistry,tianyan,test]"
```

Run the complete test suite after installation:

```bash
pytest -q
```

## Quick start: a precompiled one-parameter VQE

Use `UCCSDFactory.from_compiled` when a Pauli-generator ansatz is already
available and molecular preprocessing is intentionally not part of the runtime.
Each generator is a list of `(pauli_string, coefficient)` terms.  Pauli strings
use the package's legacy `q0-left` order: the leftmost character acts on qubit
0.

```python
from cqlib_vqe import DirectStatevectorEstimator, UCCSDFactory, VQESolver

# G = i * 0.5 * Y; the one-qubit Hamiltonian is H = Z.
factory = UCCSDFactory.from_compiled(
    n_qubits=1,
    n_electrons=0,
    generators=[[('Y', 0.5)]],
    initial_values=[0.1],
    construction_mode='jit',
)

estimator = DirectStatevectorEstimator(n_qubits=1)
solver = VQESolver(
    factory,
    estimator,
    optimizer_method='COBYLA',
    max_iter=30,
    execution_mode='auto',
)
result = solver.run([('Z', 1.0)])

print(result['optimal_value'])
print(result['optimal_params'])
```

`solver.run(...)` returns a dictionary containing `optimal_value`,
`optimal_params`, `n_evals`, `n_iters`, `success`, `message`,
`execution_mode`, and the complete energy `history`.

This smoke example deliberately caps COBYLA at 30 evaluations. It can therefore
reach the numerical minimum while still returning `success=False` because the
evaluation budget is exhausted. Increase `max_iter` and define an
energy/parameter convergence criterion appropriate to the production study.

The runnable version is available as:

```bash
python examples/minimal_compiled_smoke.py
```

## Molecular VQE workflow

For a molecular calculation, first build `MolecularDataEngine`. Its `run()`
method performs the electronic-structure preprocessing and exposes
`n_qubits`, `n_electrons`, `hamiltonian_data`, CCSD initial amplitudes, and
reference energies. Then construct the UCCSD circuit factory, choose an
estimator, and start `VQESolver`.

```python
from cqlib_vqe import (
    DirectStatevectorEstimator,
    MolecularDataEngine,
    UCCSDFactory,
    VQESolver,
)

molecule = MolecularDataEngine(
    geometry=[
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 0.735)),
    ],
    basis='sto-3g',
    multiplicity=1,
    charge=0,
    mapper_type='jw',
    excitation_threshold=1e-3,
).run()

factory = UCCSDFactory(
    molecule,
    construction_mode='jit',
    trotter_steps=1,
    trotter_order=2,
)
estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
solver = VQESolver(
    factory,
    estimator,
    optimizer_method='COBYLA',
    max_iter=100,
    tol=1e-7,
    execution_mode='auto',
)

def progress(_parameters, energy, evaluation):
    if evaluation == 1 or evaluation % 10 == 0:
        print(f'eval={evaluation:4d}  energy={energy:.12f} Ha')

result = solver.run(molecule.hamiltonian_data, callback=progress)
print('VQE energy:', result['optimal_value'])
print('reference energies:', molecule.reference_energies)
```

Run the maintained H2 example directly with:

```bash
python examples/h2_vqe_cqlib2.py
```

### Choosing an execution mode

`execution_mode='auto'` is the recommended local default. It chooses the
fused direct-statevector path when the factory and estimator support it, and
otherwise uses circuit execution. Use `execution_mode='circuit'` when a
circuit-oriented estimator is required, for example a cloud estimator. Use
`execution_mode='direct_statevector'` only when both
`factory.build_statevector(...)` and `estimator.evaluate_parameters(...)` are
available.

`construction_mode='jit'` builds numeric circuits for repeated evaluation.
`construction_mode='bind'` keeps a symbolic template and is useful when the
target execution path benefits from parameter binding.

## Mapping choice and active spaces

Set `mapper_type` to one of `"jw"`, `"bk"`, or `"parity"` when constructing
`MolecularDataEngine`:

```python
molecule = MolecularDataEngine(
    geometry=[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.735))],
    basis='sto-3g',
    mapper_type='bk',
).run()
```

The selected mapping is applied consistently to the molecular Hamiltonian,
the UCCSD excitation generators, and the Hartree-Fock reference.  In BK and
parity encodings, the reference computational-basis bit pattern is not in
general characterized by its Hamming weight alone; do not replace the factory
reference construction with a JW occupancy shortcut.

For larger systems, set an explicit active-space policy before running the
molecule engine. The qubit budget must be positive and even.

```python
from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine

active_space = ActiveSpaceConfig.automatic(
    max_active_qubits=10,
    freeze_core=True,
    min_virtual_orbitals=1,
)
molecule = MolecularDataEngine(
    geometry=[('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.596))],
    basis='sto-3g',
    mapper_type='jw',
    active_space=active_space,
).run()

print(molecule.active_space_report)
```

Use `ActiveSpaceConfig.full()` to retain the full orbital space.  The active
space report should be recorded with every numerical result because it fixes
the simulated Hamiltonian and parameter pool.

## Adaptive selected-UCCSD

The adaptive solver initializes an ansatz from canonical CCSD amplitudes and
then scans the complete remaining canonical operator pool.  Candidates with
zero initial CCSD amplitude remain eligible and are ranked by their measured
energy gradients.

The five-molecule runner provides an auditable end-to-end workflow. Start with
H2 and a small number of pool scans:

```bash
python examples/run_five_adaptive_full_pool.py h2 \\
  --output-dir run_outputs/h2_adaptive \\
  --pool-rounds 2 \\
  --maxiter 80
```

The output directory contains per-molecule logs and JSON records, plus a CSV
and JSON summary. The runner accepts `h2`, `h4`, `lih`, `beh2`, and `h2o`; omit
the positional molecule list to run all five. Use `--active-space-policy full`
only when the full orbital space is intended and the available resources are
adequate.

## Tianyan execution

Tianyan runs submit real measurement tasks and may consume platform credits.
Install the `tianyan` extra, configure credentials outside version control, and
inspect the available backend names before submitting a VQE job:

```bash
export TIANYAN_API_KEY='YOUR_API_KEY'
python tools/check_tianyan_api.py --online
```

Then select a backend and physical-qubit mapping. The mapping is ordered from
logical qubits to physical qubits and must contain exactly `n_qubits` entries.

```bash
export TIANYAN_DEVICE='BACKEND_NAME'
export TIANYAN_PHYSICAL_QUBITS='0,1,2,3'
export TIANYAN_SHOTS=2000
export TIANYAN_MAXITER=10
export TIANYAN_CALIBRATION=disabled
python -u examples/h2_vqe_tianyan.py
```

`TianyanEnergyEstimator` groups qubit-wise commuting Pauli terms before
submission and validates task ordering, measured qubit headers, and count
payloads on return. The estimator does not infer routing or silently repair
backend metadata. Never commit API keys, credential files, or raw cloud output
that contains account information.

## Useful entry points

| Task | Command |
| --- | --- |
| Minimal precompiled VQE | `python examples/minimal_compiled_smoke.py` |
| Local H2 UCCSD-VQE | `python examples/h2_vqe_cqlib2.py` |
| Adaptive H2 workflow | `python examples/run_five_adaptive_full_pool.py h2` |
| Tianyan interface check | `python tools/check_tianyan_api.py --online` |
| Full test suite | `pytest -q` |

## License

Apache License 2.0. See [LICENSE](LICENSE).
