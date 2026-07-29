# API compatibility

The canonical public namespace is `cqlib_vqe`:

```python
from cqlib_vqe import MolecularDataEngine, UCCSDFactory, VQESolver
```

For compatibility with existing 1.x users, the top-level packages `chemistry`, `vqe`, `optim`, `utils`, and `util` remain installed. Their main public objects are the same Python objects exported through `cqlib_vqe`.

The legacy aliases `UCCSD_Factory` and `TianyanCloudEnergyEstimator` remain available. New code should use `UCCSDFactory` and `TianyanEnergyEstimator`.
