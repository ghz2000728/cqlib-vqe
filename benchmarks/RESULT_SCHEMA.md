# Result schema 1.1

Top-level schema: `cqlib-vqe-benchmark/1.1`.

Each molecule may contain `engine`, `workflow`, or both.

The primary engine record is `native_pauli_objective`. Internal control records
are `decomposed_pauli_objective`, `circuit_bind_objective`,
`circuit_jit_objective`, and `assign_parameters_only`.

`speedups.native_vs_decomposed` is defined as:

```text
decomposed median seconds / native median seconds
```

Values greater than one mean the native Pauli kernel is faster.

Workflow records preserve the 1.0 fields and add:

- `execution_profile`
- `requested_execution_mode`
- `candidate_initialization`
