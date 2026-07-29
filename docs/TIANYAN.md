# Tianyan execution contract

## Public names

The adapter follows the standalone SDK exactly:

- Python package: `cqlib_tianyan`
- Platform: `TianyanPlatform`
- Backend: `TianyanBackend`
- Task handle: `TaskHandle`
- VQE estimator: `TianyanEnergyEstimator`

Backend identifiers are returned by `TianyanPlatform.list_backends()` and are case-sensitive. The package does not assume a particular hardware or simulator name.

## Submission pipeline

For every energy evaluation, the estimator:

1. validates the Hamiltonian;
2. separates the identity contribution;
3. forms QWC groups, or one circuit per term with `grouping="none"`;
4. compiles the base cqlib circuit to the Tianyan-native basis;
5. remaps logical qubits to the explicitly supplied physical mapping;
6. appends native X/Y measurement-basis rotations and `M` instructions;
7. rejects any remaining non-native instruction;
8. calls `TianyanBackend.run_with_mode`;
9. requires exact task order, exact result-qubit headers, and non-empty counts;
10. reconstructs Pauli expectations and adds the identity energy.

Accepted QCIS instructions are `RZ`, `X2P`, `X2M`, `Y2P`, `Y2M`, `XY2P`, `XY2M`, `CZ`, and `M`.

## Deliberate fail-fast behavior

The estimator does not:

- accept `calibration_mode="auto"`;
- reorder task results;
- synthesize missing result-qubit headers;
- substitute probabilities when counts are empty;
- fall back to `run()` or `run_raw()`;
- invent a routing solution.

## Validation sequence

```bash
python tools/check_tianyan_api.py --online
python tools/check_tianyan_api.py --online --device BACKEND_NAME
python examples/tianyan_pauli_strict_probe.py --device BACKEND_NAME --qubit 0 --shots 1000
python examples/tianyan_qwc_compare.py --device BACKEND_NAME --qubits 0,1 --shots 1000
```

Then run the grouped-energy smoke test and H2 example. Start with low shots and a small optimization budget.

## Credentials

Use `TIANYAN_API_KEY` or the SDK credential store. Do not pass API keys as command-line arguments, commit them to source control, or include them in logs.
