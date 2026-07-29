# cqlib-vqe

`cqlib-vqe` provides active-space molecular Hamiltonians, canonical singlet-UCCSD circuits, adaptive operator-pool selection, high-performance statevector objectives, and a strict Tianyan cloud/hardware estimator for cqlib 2.x.

## Highlights

- Canonical closed-shell singlet-UCCSD with auditable packed parameter ordering.
- End-to-end Jordan-Wigner, Bravyi-Kitaev, and parity encodings: the molecular
  Hamiltonian, canonical excitation generators, and Hartree-Fock reference
  state always use the same mapping.
- Full operator-pool adaptive screening, including initially zero-amplitude generators.
- Native `Statevector.apply_pauli_rotation` execution when supported by cqlib.
- Content-based Hamiltonian caching and reusable compiled Pauli schedules.
- QWC-grouped Tianyan measurements with strict task, qubit-header, and counts validation.
- Tianyan-native QCIS compilation (`RZ`, `X2P/X2M`, `Y2P/Y2M`, `XY2P/XY2M`, `CZ`, `M`) before cloud submission.
- A canonical `cqlib_vqe` import namespace; legacy `vqe` and `chemistry` imports remain available throughout the 1.x series.

## Installation

Core package:

```bash
pip install .
```

Quantum chemistry dependencies:

```bash
pip install ".[chemistry]"
```

Tianyan support:

```bash
pip install ".[chemistry,tianyan]"
```

Development installation:

```bash
pip install -e ".[chemistry,tianyan,test]"
pytest -q
```

## Local H2 VQE

```python
from cqlib_vqe import (
    DirectStatevectorEstimator,
    MolecularDataEngine,
    UCCSDFactory,
    VQESolver,
)

molecule = MolecularDataEngine(
    geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
    basis="sto-3g",
    mapper_type="jw",
).run()

factory = UCCSDFactory(molecule, construction_mode="jit")
estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
solver = VQESolver(factory, estimator, optimizer_method="COBYLA", max_iter=100)
result = solver.run(molecule.hamiltonian_data)
print(result["optimal_value"])
```

Set ``mapper_type`` to ``"jw"``, ``"bk"``, or ``"parity"``.  The factory
encodes the Hartree-Fock reference determinant with the selected binary code;
BK and parity do not generally have a computational-basis Hamming weight equal
to the electron count.

## Tianyan cloud and hardware

Discover the backend names available to the authenticated account:

```bash
export TIANYAN_API_KEY='YOUR_API_KEY'
python tools/check_tianyan_api.py --online
```

Run H2 on a selected backend:

```bash
export TIANYAN_DEVICE='BACKEND_NAME_FROM_THE_LIST'
export TIANYAN_PHYSICAL_QUBITS='0,1,2,3'
export TIANYAN_SHOTS=2000
export TIANYAN_MAXITER=10
export TIANYAN_CALIBRATION=disabled
python -u examples/h2_vqe_tianyan.py
```

Backend names are case-sensitive and are not hard-coded by this package. Physical-qubit mappings must be valid for the selected backend topology; the estimator does not invent routing or silently repair result metadata.

See [docs/TIANYAN.md](docs/TIANYAN.md) for the validation sequence and execution contract.

## Validation

The release includes unit tests for active-space selection, canonical pool construction, statevector execution, adaptive full-pool behavior, QWC grouping, Tianyan-native QCIS generation, task ordering, result-qubit ordering, and canonical/legacy import compatibility.

A manual cloud full-amplitude H2 integration run returned `-1.136986799379 Ha` against an FCI reference of `-1.137306035753 Ha` (absolute error `0.319 mHa`), validating the end-to-end QCIS, grouping, result parsing, and energy reconstruction path.

See [docs/VALIDATION.md](docs/VALIDATION.md).

## Security

Never commit API keys, credential files, raw cloud responses containing account metadata, or shell history containing secrets. See [SECURITY.md](SECURITY.md).

## License

Apache License 2.0.
