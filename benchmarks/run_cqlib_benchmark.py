#!/usr/bin/env python3
"""Reproducible cqlib VQE benchmark suite.

The suite produces one self-contained JSON document that can later be joined
with Qiskit Nature and PennyLane results.  It has two independent tracks:

``engine``
    Same cqlib ansatz and Hamiltonian, repeated parameter updates.  The track
    separates symbolic-template construction, ``Circuit.assign_parameters``,
    bound-circuit execution, numeric JIT circuit construction, and the direct
    statevector path.

``workflow``
    End-to-end cqlib algorithm-package ablations: full canonical UCCSD,
    static CCSD-amplitude selection, and one/two complete operator-pool scans.
    FCI is never supplied to the solver as a stopping criterion; it is used
    only after a run to label the final error.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import socket
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cqlib import Circuit  # noqa: E402
from cqlib.qis import Statevector  # noqa: E402
from cqlib.ir import qcis  # noqa: E402
from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine  # noqa: E402
from cqlib_vqe import (  # noqa: E402
    AdaptiveSelectedUCCSDSolver,
    AdaptiveSelectionConfig,
    DirectStatevectorEstimator,
    NativeStatevectorEstimator,
    UCCSDFactory,
    VQESolver,
)

SCHEMA_VERSION = "cqlib-vqe-benchmark/1.1"
CHEMICAL_ACCURACY_HA = 1.6e-3
DEFAULT_MOLECULES = ("h2", "h4", "lih", "beh2", "h2o")
DEFAULT_VARIANTS = ("full_uccsd", "static_selected", "adaptive_r1", "adaptive_r2")
ALLOWED_VARIANTS = (*DEFAULT_VARIANTS, "adaptive_custom")
ENGINE_MODES = (
    "assign_parameters_only",
    "circuit_bind_objective",
    "circuit_jit_objective",
    "native_pauli_objective",
    "decomposed_pauli_objective",
)

MOLECULE_CASES: dict[str, dict[str, Any]] = {
    "h2": {
        "geometry": [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))],
        "active_qubits": None,
    },
    "h4": {
        "geometry": [
            ("H", (0.0, 0.0, -1.5)),
            ("H", (0.0, 0.0, -0.5)),
            ("H", (0.0, 0.0, 0.5)),
            ("H", (0.0, 0.0, 1.5)),
        ],
        "active_qubits": None,
    },
    "lih": {
        "geometry": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.596))],
        "active_qubits": 10,
    },
    "beh2": {
        "geometry": [
            ("H", (0.0, 0.0, -1.3264)),
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.3264)),
        ],
        "active_qubits": 12,
    },
    "h2o": {
        "geometry": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.587)),
            ("H", (0.0, -0.757, 0.587)),
        ],
        "active_qubits": 12,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return repr(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
    }


def parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def environment_metadata() -> dict[str, Any]:
    cpu_count = os.cpu_count()
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        rss = int(process.memory_info().rss)
        physical_cores = psutil.cpu_count(logical=False)
        cpu_name = platform.processor() or None
    except Exception:
        rss = None
        physical_cores = None
        cpu_name = platform.processor() or None
    return {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_name,
        "python": sys.version,
        "python_executable": sys.executable,
        "logical_cpu_count": cpu_count,
        "physical_cpu_count": physical_cores,
        "process_rss_bytes_at_start": rss,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "packages": {
            name: package_version(name)
            for name in (
                "cqlib",
                "cqlib-vqe",
                "numpy",
                "scipy",
                "pyscf",
                "openfermion",
                "openfermionpyscf",
            )
        },
        "sources": {
            "project_root": str(PROJECT_ROOT),
            "uccsd_factory": inspect.getsourcefile(UCCSDFactory),
            "adaptive_solver": inspect.getsourcefile(AdaptiveSelectedUCCSDSolver),
        },
    }


def active_space(case: Mapping[str, Any], policy: str) -> ActiveSpaceConfig:
    if policy == "full" or case["active_qubits"] is None:
        return ActiveSpaceConfig.full()
    return ActiveSpaceConfig.automatic(
        max_active_qubits=int(case["active_qubits"]), freeze_core=True
    )


def prepare_molecule(name: str, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    case = MOLECULE_CASES[name]
    started = time.perf_counter()
    molecule = MolecularDataEngine(
        geometry=case["geometry"],
        basis=args.basis,
        multiplicity=1,
        charge=0,
        mapper_type="jw",
        excitation_threshold=args.initial_threshold,
        active_space=active_space(case, args.active_space_policy),
    ).run(run_fci=True, audit_ccsd_basis=True)
    elapsed = time.perf_counter() - started

    estimator = NativeStatevectorEstimator(n_qubits=molecule.n_qubits)
    hf_circuit = Circuit(molecule.n_qubits)
    for qubit in range(molecule.n_electrons):
        hf_circuit.x(qubit)
    mapped_hf = float(estimator.evaluate(hf_circuit, molecule.hamiltonian_data))
    hf_delta = mapped_hf - float(molecule.hf_energy)
    if abs(hf_delta) > 1e-8:
        raise AssertionError(
            f"Mapped HF energy mismatch for {name}: {hf_delta:+.6e} Ha"
        )

    amplitudes = canonical_amplitude_map(molecule)
    initial = initial_indices(amplitudes, args.initial_threshold)
    report = molecule.active_space_report
    metadata = {
        "name": name,
        "geometry": case["geometry"],
        "basis": args.basis,
        "mapper": "jordan_wigner",
        "preprocessing_seconds": elapsed,
        "n_qubits": int(molecule.n_qubits),
        "n_electrons": int(molecule.n_electrons),
        "hamiltonian_term_count": len(molecule.hamiltonian_data),
        "full_canonical_pool_count": int(molecule.ccsd_full_parameter_count),
        "initial_selected_count": len(initial),
        "initial_selected_indices": initial,
        "zero_ccsd_amplitude_indices": [
            index
            for index, amplitude in sorted(amplitudes.items())
            if abs(amplitude) <= args.zero_tolerance
        ],
        "energies_ha": {
            "hf": float(molecule.hf_energy),
            "ccsd": float(molecule.ccsd_energy),
            "fci": float(molecule.fci_energy),
            "mapped_hf": mapped_hf,
            "mapped_hf_minus_pyscf_hf": hf_delta,
        },
        "active_space": asdict(report) if report is not None else None,
    }
    return molecule, metadata


def canonical_amplitude_map(molecule: Any) -> dict[int, float]:
    descriptors = list(molecule.ccsd_descriptors)
    values = np.asarray(molecule.ccsd_parameters, dtype=float)
    if len(descriptors) != values.size:
        raise ValueError("Canonical descriptor/amplitude count mismatch")
    result = {
        int(descriptor.packed_index): float(value)
        for descriptor, value in zip(descriptors, values, strict=True)
    }
    expected = set(range(int(molecule.ccsd_full_parameter_count)))
    if set(result) != expected:
        raise ValueError("Canonical amplitude map is not the complete packed pool")
    return result


def initial_indices(amplitudes: Mapping[int, float], threshold: float) -> list[int]:
    selected = [index for index, value in amplitudes.items() if abs(value) >= threshold]
    if not selected:
        selected = [max(amplitudes, key=lambda index: abs(amplitudes[index]))]
    return sorted(selected)


def vector_for_indices(
    indices: Sequence[int], amplitudes: Mapping[int, float]
) -> np.ndarray:
    return np.asarray([amplitudes[int(index)] for index in indices], dtype=float)


def rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def time_repeated(
    operation: Callable[[np.ndarray], Any],
    vectors: Sequence[np.ndarray],
    *,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    if not vectors:
        raise ValueError("vectors must not be empty")
    for index in range(warmup):
        operation(vectors[index % len(vectors)])

    totals: list[float] = []
    checksums: list[float] = []
    rss_before = rss_bytes()
    gc_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(repeats):
            checksum = 0.0
            started = time.perf_counter_ns()
            for vector in vectors:
                value = operation(vector)
                if isinstance(value, (float, int, np.floating)):
                    checksum += float(value)
                elif hasattr(value, "num_qubits"):
                    checksum += float(value.num_qubits)
            totals.append((time.perf_counter_ns() - started) / 1e9)
            checksums.append(checksum)
    finally:
        if gc_enabled:
            gc.enable()
    rss_after = rss_bytes()
    per_eval = [total / len(vectors) for total in totals]
    return {
        "evaluations_per_repeat": len(vectors),
        "repeats": repeats,
        "warmup_evaluations": warmup,
        "total_seconds_per_repeat": totals,
        "seconds_per_evaluation": distribution(per_eval),
        "evaluations_per_second": distribution(
            [len(vectors) / total for total in totals if total > 0]
        ),
        "checksum_per_repeat": checksums,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
    }


def circuit_metrics(circuit: Any) -> dict[str, Any]:
    gate_counts: Counter[str] = Counter()
    one_qubit = 0
    two_qubit = 0
    multi_qubit = 0
    for operation in circuit.operations:
        name = "unknown"
        value_instruction = operation.instruction
        if getattr(value_instruction, "is_instruction", False):
            instruction = value_instruction.instruction
            if instruction is not None:
                name = str(instruction.name)
        elif getattr(value_instruction, "is_classical_control", False):
            name = "classical_control"
        qubit_count = len(operation.qubits)
        gate_counts[name] += 1
        if qubit_count == 1:
            one_qubit += 1
        elif qubit_count == 2:
            two_qubit += 1
        elif qubit_count > 2:
            multi_qubit += 1
    try:
        qcis_text = qcis.dumps(circuit)
    except Exception:
        qcis_text = ""
    return {
        "num_qubits": int(circuit.num_qubits),
        "operation_count": len(circuit.operations),
        "depth": int(circuit.depth()),
        "one_qubit_operation_count": one_qubit,
        "two_qubit_operation_count": two_qubit,
        "multi_qubit_operation_count": multi_qubit,
        "gate_counts": dict(sorted(gate_counts.items())),
        "qcis_line_count": len(qcis_text.splitlines()) if qcis_text else None,
        "qcis_character_count": len(qcis_text) if qcis_text else None,
    }


def parameter_vectors(
    initial: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    perturbation: float,
) -> list[np.ndarray]:
    if initial.size == 0:
        return [initial.copy() for _ in range(count)]
    return [
        initial + rng.normal(loc=0.0, scale=perturbation, size=initial.size)
        for _ in range(count)
    ]


def force_decomposed_statevector(factory: UCCSDFactory) -> None:
    """Disable only the fused Rust Pauli kernel for an internal control path.

    The compiled generator order, Trotter schedule, Hamiltonian and statevector
    backend remain identical.  Only each Pauli exponential is executed through
    the legacy H/RX/CX/RZ decomposition.
    """

    plans = getattr(factory, "_statevector_plans", None)
    if plans is None:
        raise AttributeError("UCCSDFactory does not expose compiled statevector plans")
    factory._statevector_plans = tuple(  # type: ignore[attr-defined]
        tuple(replace(rotation, native_pauli=None) for rotation in plan)
        for plan in plans
    )
    factory._native_pauli_rotation_available = False  # type: ignore[attr-defined]


def run_engine_track(
    molecule: Any,
    molecule_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    amplitudes = canonical_amplitude_map(molecule)
    selected = initial_indices(amplitudes, args.initial_threshold)
    initial = vector_for_indices(selected, amplitudes)
    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
    if hasattr(estimator, "prepare_hamiltonian"):
        estimator.prepare_hamiltonian(
            molecule.hamiltonian_data,
            expected_num_qubits=molecule.n_qubits,
        )

    started = time.perf_counter()
    bind_factory = UCCSDFactory(
        molecule,
        selected_packed_indices=selected,
        construction_mode="bind",
        trotter_order=args.trotter_order,
        trotter_steps=args.trotter_steps,
    )
    bind_factory_build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    native_factory = UCCSDFactory(
        molecule,
        selected_packed_indices=selected,
        construction_mode="jit",
        trotter_order=args.trotter_order,
        trotter_steps=args.trotter_steps,
    )
    native_factory_build_seconds = time.perf_counter() - started

    if args.require_native_pauli and not native_factory.native_pauli_rotation_available:
        raise RuntimeError(
            "cqlib-vqe 1.4 benchmark requires "
            "Statevector.apply_pauli_rotation. Rebuild the patched cqlib2 "
            "Python binding with `maturin develop --release`."
        )

    started = time.perf_counter()
    decomposed_factory = UCCSDFactory(
        molecule,
        selected_packed_indices=selected,
        construction_mode="jit",
        trotter_order=args.trotter_order,
        trotter_steps=args.trotter_steps,
    )
    force_decomposed_statevector(decomposed_factory)
    decomposed_factory_build_seconds = time.perf_counter() - started

    reference_vector = initial.copy()
    circuit_bind_energy = float(
        estimator.evaluate(bind_factory.build(reference_vector), molecule.hamiltonian_data)
    )
    circuit_jit_energy = float(
        estimator.evaluate(native_factory.build(reference_vector), molecule.hamiltonian_data)
    )
    native_energy = float(
        estimator.evaluate_parameters(
            native_factory, reference_vector, molecule.hamiltonian_data
        )
    )
    decomposed_energy = float(
        estimator.evaluate_parameters(
            decomposed_factory, reference_vector, molecule.hamiltonian_data
        )
    )
    energies = (
        circuit_bind_energy,
        circuit_jit_energy,
        native_energy,
        decomposed_energy,
    )
    max_energy_delta = max(
        abs(first - second)
        for index, first in enumerate(energies)
        for second in energies[index + 1 :]
    )
    if max_energy_delta > 1e-9:
        raise AssertionError(
            f"Engine-path energy mismatch for {molecule_name}: "
            f"{max_energy_delta:.3e} Ha"
        )

    rng = np.random.default_rng(args.seed + sum(ord(char) for char in molecule_name))
    batches: dict[str, Any] = {}
    for count in args.engine_counts:
        vectors = parameter_vectors(
            initial,
            count,
            rng=rng,
            perturbation=args.parameter_perturbation,
        )
        mode_results: dict[str, Any] = {}
        mode_results["assign_parameters_only"] = time_repeated(
            lambda values: bind_factory.build(values),
            vectors,
            repeats=args.engine_repeats,
            warmup=args.engine_warmup,
        )
        mode_results["circuit_bind_objective"] = time_repeated(
            lambda values: estimator.evaluate(
                bind_factory.build(values), molecule.hamiltonian_data
            ),
            vectors,
            repeats=args.engine_repeats,
            warmup=args.engine_warmup,
        )
        mode_results["circuit_jit_objective"] = time_repeated(
            lambda values: estimator.evaluate(
                native_factory.build(values), molecule.hamiltonian_data
            ),
            vectors,
            repeats=args.engine_repeats,
            warmup=args.engine_warmup,
        )
        mode_results["native_pauli_objective"] = time_repeated(
            lambda values: estimator.evaluate_parameters(
                native_factory, values, molecule.hamiltonian_data
            ),
            vectors,
            repeats=args.engine_repeats,
            warmup=args.engine_warmup,
        )
        mode_results["decomposed_pauli_objective"] = time_repeated(
            lambda values: estimator.evaluate_parameters(
                decomposed_factory, values, molecule.hamiltonian_data
            ),
            vectors,
            repeats=args.engine_repeats,
            warmup=args.engine_warmup,
        )

        medians = {
            mode: mode_results[mode]["seconds_per_evaluation"]["median"]
            for mode in ENGINE_MODES
        }
        native_median = medians["native_pauli_objective"]
        mode_results["speedups"] = {
            "native_vs_decomposed": (
                None
                if not native_median
                else medians["decomposed_pauli_objective"] / native_median
            ),
            "native_vs_circuit_bind": (
                None
                if not native_median
                else medians["circuit_bind_objective"] / native_median
            ),
            "native_vs_circuit_jit": (
                None
                if not native_median
                else medians["circuit_jit_objective"] / native_median
            ),
        }
        batches[str(count)] = mode_results
        print(
            f"[engine] {molecule_name:4s} count={count:4d} "
            f"native={native_median:.6e}s "
            f"decomp={medians['decomposed_pauli_objective']:.6e}s "
            f"bind={medians['circuit_bind_objective']:.6e}s "
            f"speedup={mode_results['speedups']['native_vs_decomposed']:.3f}x"
        )

    symbolic_template = getattr(bind_factory, "_template", None)
    template_metrics = (
        circuit_metrics(symbolic_template) if symbolic_template is not None else None
    )
    bound_metrics = circuit_metrics(bind_factory.build(initial))
    return {
        "ansatz": "initial_ccsd_selected",
        "recommended_case": "native_pauli_objective",
        "selected_packed_indices": selected,
        "parameter_count": len(selected),
        "pauli_rotation_term_count": sum(
            len(generator.terms) for generator in bind_factory.generators
        ),
        "native_pauli_rotation_available": bool(
            native_factory.native_pauli_rotation_available
        ),
        "factory_construction_seconds": {
            "bind_symbolic_template": bind_factory_build_seconds,
            "native_compiled_plan": native_factory_build_seconds,
            "decomposed_compiled_plan": decomposed_factory_build_seconds,
        },
        "assign_parameters_api_used": True,
        "energy_consistency": {
            "circuit_bind_energy_ha": circuit_bind_energy,
            "circuit_jit_energy_ha": circuit_jit_energy,
            "native_pauli_energy_ha": native_energy,
            "decomposed_pauli_energy_ha": decomposed_energy,
            "max_absolute_delta_ha": max_energy_delta,
        },
        "symbolic_template_resources": template_metrics,
        "bound_circuit_resources": bound_metrics,
        "batches": batches,
    }


@contextlib.contextmanager
def hide_reference_energy(molecule: Any) -> Iterator[float | None]:
    """Prevent AdaptiveSelectedUCCSDSolver from using FCI as a stop signal."""

    original = getattr(molecule, "fci_energy", None)
    molecule.fci_energy = None
    try:
        yield original
    finally:
        molecule.fci_energy = original


def workflow_resource_metrics(
    molecule: Any,
    indices: Sequence[int],
    parameters: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    factory = UCCSDFactory(
        molecule,
        selected_packed_indices=indices,
        construction_mode="bind",
        trotter_order=args.trotter_order,
        trotter_steps=args.trotter_steps,
    )
    circuit = factory.build(np.asarray(parameters, dtype=float))
    metrics = circuit_metrics(circuit)
    metrics.update(
        {
            "parameter_count": len(indices),
            "selected_packed_indices": list(indices),
            "pauli_rotation_term_count": sum(
                len(generator.terms) for generator in factory.generators
            ),
        }
    )
    return metrics


def one_workflow_run(
    molecule: Any,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
    amplitudes = canonical_amplitude_map(molecule)
    fci = float(molecule.fci_energy)
    started = time.perf_counter()

    if variant == "full_uccsd":
        indices = list(range(int(molecule.ccsd_full_parameter_count)))
        factory = UCCSDFactory(
            molecule,
            selected_packed_indices=indices,
            construction_mode=args.construction_mode,
            trotter_order=args.trotter_order,
            trotter_steps=args.trotter_steps,
        )
        initial = vector_for_indices(indices, amplitudes)
        solver = VQESolver(
            factory,
            estimator,
            optimizer_method="COBYLA",
            max_iter=args.maxiter,
            tol=args.optimizer_tolerance,
            execution_mode=args.execution_mode,
            optimizer_options={"rhobeg": args.rhobeg, "catol": 1e-8},
        )
        result = solver.run(molecule.hamiltonian_data, initial_params=initial)
        selected = indices
        optimal_params = np.asarray(result["optimal_params"], dtype=float)
        optimal_value = float(result["optimal_value"])
        total_evaluations = int(result["n_evals"])
        optimizer_evaluations = int(result["n_evals"])
        gradient_evaluations = 0
        pool_scans = 0
        stop_reason = str(result["message"])
        rounds: list[dict[str, Any]] = []
    else:
        pool_rounds = {
            "static_selected": 0,
            "adaptive_r1": 1,
            "adaptive_r2": 2,
            "adaptive_custom": int(args.pool_rounds),
        }[variant]
        config = AdaptiveSelectionConfig(
            initial_amplitude_threshold=args.initial_threshold,
            candidate_amplitude_threshold=0.0,
            pool_rounds=pool_rounds,
            max_add_per_round=args.max_add_per_round,
            max_parameters=args.max_parameters,
            gradient_delta=args.gradient_delta,
            gradient_threshold=args.gradient_threshold,
            # This value is inert because FCI is hidden during the run.
            target_energy_error=CHEMICAL_ACCURACY_HA,
            optimizer_maxiter=args.maxiter,
            optimizer_tolerance=args.optimizer_tolerance,
            optimizer_options={"rhobeg": args.rhobeg, "catol": 1e-8},
            construction_mode=args.construction_mode,
            execution_mode=args.execution_mode,
            trotter_steps=args.trotter_steps,
            trotter_order=args.trotter_order,
            candidate_initialization=args.candidate_initialization,
        )
        adaptive = AdaptiveSelectedUCCSDSolver(molecule, estimator, config)
        with hide_reference_energy(molecule):
            result = adaptive.run(reference_energy=None)
        selected = list(result.selected_packed_indices)
        optimal_params = np.asarray(result.optimal_params, dtype=float)
        optimal_value = float(result.optimal_value)
        total_evaluations = int(result.total_evaluations)
        optimizer_evaluations = sum(
            int(round_item.optimizer_evaluations) for round_item in result.rounds
        )
        gradient_evaluations = total_evaluations - optimizer_evaluations
        pool_scans = int(result.pool_scans_completed)
        stop_reason = str(result.stop_reason)
        rounds = [
            {
                "round_index": item.round_index,
                "pool_scan_index": item.pool_scan_index,
                "selected_before": list(item.selected_before),
                "candidate_pool_count": len(item.candidate_pool_before),
                "optimized_energy_ha": item.optimized_energy,
                "optimizer_evaluations": item.optimizer_evaluations,
                "gradient_candidate_count": len(item.candidates),
                "gradient_norm": item.gradient_norm,
                "eligible_indices": list(item.eligible_indices),
                "added_indices": list(item.added_indices),
                "selected_after": list(item.selected_after),
                "wall_time_seconds": item.wall_time_seconds,
                "top_candidate_gradients": [
                    {
                        "packed_index": candidate.packed_index,
                        "gradient": candidate.gradient,
                        "absolute_gradient": candidate.absolute_gradient,
                        "ccsd_amplitude": candidate.ccsd_amplitude,
                    }
                    for candidate in item.candidates[: args.record_top_gradients]
                ],
            }
            for item in result.rounds
        ]

    elapsed = time.perf_counter() - started
    final_error = optimal_value - fci
    zero_tolerance = args.zero_tolerance
    zero_added = [
        index for index in selected if abs(amplitudes[index]) <= zero_tolerance
    ]
    resources = workflow_resource_metrics(
        molecule, selected, optimal_params, args
    )
    return {
        "variant": variant,
        "wall_time_seconds": elapsed,
        "optimal_energy_ha": optimal_value,
        "fci_energy_ha": fci,
        "signed_fci_error_ha": final_error,
        "absolute_fci_error_mha": abs(final_error) * 1000.0,
        "chemical_accuracy": abs(final_error) <= CHEMICAL_ACCURACY_HA,
        "selected_parameter_count": len(selected),
        "full_parameter_count": int(molecule.ccsd_full_parameter_count),
        "parameter_compression_fraction": 1.0
        - len(selected) / int(molecule.ccsd_full_parameter_count),
        "selected_packed_indices": selected,
        "zero_ccsd_amplitude_selected_indices": zero_added,
        "total_energy_evaluations": total_evaluations,
        "optimizer_energy_evaluations": optimizer_evaluations,
        "candidate_gradient_energy_evaluations": gradient_evaluations,
        "pool_scans_completed": pool_scans,
        "stop_reason": stop_reason,
        "fci_used_for_stopping": False,
        "execution_profile": "native_pauli_auto",
        "requested_execution_mode": args.execution_mode,
        "candidate_initialization": args.candidate_initialization,
        "rounds": rounds,
        "circuit_resources": resources,
    }


def aggregate_workflow_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run.get("status") == "ok"]
    return {
        "run_count": len(runs),
        "successful_run_count": len(successful),
        "chemical_accuracy_success_count": sum(
            bool(run["chemical_accuracy"]) for run in successful
        ),
        "wall_time_seconds": distribution(
            [float(run["wall_time_seconds"]) for run in successful]
        ),
        "absolute_fci_error_mha": distribution(
            [float(run["absolute_fci_error_mha"]) for run in successful]
        ),
        "total_energy_evaluations": distribution(
            [float(run["total_energy_evaluations"]) for run in successful]
        ),
        "selected_parameter_count": distribution(
            [float(run["selected_parameter_count"]) for run in successful]
        ),
    }


def run_workflow_track(
    molecule: Any,
    molecule_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in args.variants:
        runs: list[dict[str, Any]] = []
        for repeat in range(args.workflow_repeats):
            print(
                f"[workflow] {molecule_name:4s} variant={variant:15s} "
                f"repeat={repeat + 1}/{args.workflow_repeats}"
            )
            try:
                run = one_workflow_run(molecule, variant, args)
                run.update({"status": "ok", "repeat_index": repeat})
            except Exception as exc:
                run = {
                    "status": "failed",
                    "repeat_index": repeat,
                    "variant": variant,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                if args.fail_fast:
                    raise
            runs.append(run)
        variants[variant] = {
            "runs": runs,
            "aggregate": aggregate_workflow_runs(runs),
        }
    return {"variants": variants}


def flat_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for molecule_name, molecule in payload.get("molecules", {}).items():
        if molecule.get("status") != "ok":
            continue
        engine = molecule.get("engine")
        if engine:
            for count, batch in engine.get("batches", {}).items():
                for mode in ENGINE_MODES:
                    item = batch.get(mode)
                    if not item:
                        continue
                    records.append(
                        {
                            "framework": "cqlib",
                            "track": "engine",
                            "molecule": molecule_name,
                            "case": mode,
                            "evaluation_batch_size": int(count),
                            "parameter_count": engine["parameter_count"],
                            "median_seconds_per_evaluation": item[
                                "seconds_per_evaluation"
                            ]["median"],
                            "median_evaluations_per_second": item[
                                "evaluations_per_second"
                            ]["median"],
                        }
                    )
        workflow = (molecule.get("workflow") or {}).get("variants", {})
        for variant, variant_data in workflow.items():
            for run in variant_data.get("runs", []):
                records.append(
                    {
                        "framework": "cqlib",
                        "track": "workflow",
                        "molecule": molecule_name,
                        "case": variant,
                        **{
                            key: run.get(key)
                            for key in (
                                "status",
                                "repeat_index",
                                "wall_time_seconds",
                                "optimal_energy_ha",
                                "absolute_fci_error_mha",
                                "chemical_accuracy",
                                "selected_parameter_count",
                                "full_parameter_count",
                                "total_energy_evaluations",
                                "optimizer_energy_evaluations",
                                "candidate_gradient_energy_evaluations",
                                "pool_scans_completed",
                            )
                        },
                    }
                )
    return records


def summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    workflow_summary: dict[str, Any] = {}
    for variant in payload["configuration"]["workflow"]["variants"]:
        runs: list[Mapping[str, Any]] = []
        for molecule in payload.get("molecules", {}).values():
            variant_data = (
                (molecule.get("workflow") or {})
                .get("variants", {})
                .get(variant, {})
            )
            runs.extend(variant_data.get("runs", []))
        successful = [run for run in runs if run.get("status") == "ok"]
        workflow_summary[variant] = {
            "molecule_run_count": len(runs),
            "successful_run_count": len(successful),
            "chemical_accuracy_success_count": sum(
                bool(run.get("chemical_accuracy")) for run in successful
            ),
            "median_wall_time_seconds": (
                statistics.median(float(run["wall_time_seconds"]) for run in successful)
                if successful
                else None
            ),
            "median_absolute_fci_error_mha": (
                statistics.median(
                    float(run["absolute_fci_error_mha"]) for run in successful
                )
                if successful
                else None
            ),
            "median_selected_parameter_count": (
                statistics.median(
                    int(run["selected_parameter_count"]) for run in successful
                )
                if successful
                else None
            ),
        }
    return {"workflow_by_variant": workflow_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecules", nargs="*", default=None)
    parser.add_argument(
        "--tracks",
        type=parse_csv_strings,
        default=["engine", "workflow"],
        help="Comma-separated: engine,workflow",
    )
    parser.add_argument(
        "--variants",
        type=parse_csv_strings,
        default=list(DEFAULT_VARIANTS),
        help="Comma-separated workflow variants",
    )
    parser.add_argument("--active-space-policy", choices=("default", "full"), default="default")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--initial-threshold", type=float, default=1e-3)
    parser.add_argument("--zero-tolerance", type=float, default=1e-14)
    parser.add_argument("--pool-rounds", type=int, default=2)
    parser.add_argument("--gradient-threshold", type=float, default=1e-3)
    parser.add_argument("--gradient-delta", type=float, default=1e-3)
    parser.add_argument("--max-add-per-round", type=int, default=2)
    parser.add_argument("--max-parameters", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-6)
    parser.add_argument("--rhobeg", type=float, default=0.03)
    parser.add_argument("--construction-mode", choices=("jit", "bind"), default="bind")
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "circuit", "direct_statevector"),
        default="auto",
    )
    parser.add_argument(
        "--candidate-initialization",
        choices=("zero", "best_probe"),
        default="best_probe",
    )
    parser.add_argument(
        "--require-native-pauli",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if cqlib2 Statevector.apply_pauli_rotation is unavailable.",
    )
    parser.add_argument("--trotter-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--trotter-steps", type=int, default=2)
    parser.add_argument("--record-top-gradients", type=int, default=10)

    parser.add_argument("--engine-counts", type=parse_csv_ints, default=[1, 10, 100])
    parser.add_argument("--engine-repeats", type=int, default=5)
    parser.add_argument("--engine-warmup", type=int, default=5)
    parser.add_argument("--parameter-perturbation", type=float, default=0.02)
    parser.add_argument("--workflow-repeats", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.molecules:
        args.molecules = list(DEFAULT_MOLECULES)
    unknown_molecules = sorted(set(args.molecules) - set(MOLECULE_CASES))
    if unknown_molecules:
        raise ValueError(f"Unknown molecules: {unknown_molecules}")
    unknown_tracks = sorted(set(args.tracks) - {"engine", "workflow"})
    if unknown_tracks:
        raise ValueError(f"Unknown tracks: {unknown_tracks}")
    unknown_variants = sorted(set(args.variants) - set(ALLOWED_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown workflow variants: {unknown_variants}")
    for name in (
        "engine_repeats",
        "workflow_repeats",
        "maxiter",
        "trotter_steps",
        "record_top_gradients",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.engine_warmup < 0:
        raise ValueError("engine_warmup must be non-negative")
    if args.initial_threshold < 0 or args.zero_tolerance < 0:
        raise ValueError("amplitude thresholds must be non-negative")
    if args.gradient_threshold < 0 or args.gradient_delta <= 0:
        raise ValueError("invalid gradient threshold/delta")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    native_available = callable(getattr(Statevector, "apply_pauli_rotation", None))
    if args.require_native_pauli and not native_available:
        raise RuntimeError(
            "Native PauliRotation is unavailable. Apply the minimal cqlib2 patch "
            "and rebuild with `maturin develop --release`, or pass "
            "--no-require-native-pauli for a compatibility-only run."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (
        PROJECT_ROOT / "run_outputs" / f"cqlib_benchmark_{timestamp}" / "cqlib_results.json"
    )
    output = output.resolve()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "framework": "cqlib",
        "benchmark_profile": "cqlib-vqe-1.4-native-pauli",
        "status": "running",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "environment": environment_metadata(),
        "configuration": {
            "molecules": list(args.molecules),
            "tracks": list(args.tracks),
            "seed": args.seed,
            "active_space_policy": args.active_space_policy,
            "basis": args.basis,
            "chemistry": {
                "initial_amplitude_threshold": args.initial_threshold,
                "zero_amplitude_tolerance": args.zero_tolerance,
                "fci_used_for_post_run_scoring_only": True,
                "chemical_accuracy_ha": CHEMICAL_ACCURACY_HA,
            },
            "engine": {
                "counts": args.engine_counts,
                "repeats": args.engine_repeats,
                "warmup": args.engine_warmup,
                "parameter_perturbation": args.parameter_perturbation,
                "modes": list(ENGINE_MODES),
                "recommended_case": "native_pauli_objective",
                "native_pauli_rotation_available": native_available,
            },
            "workflow": {
                "variants": list(args.variants),
                "repeats": args.workflow_repeats,
                "pool_rounds_default": args.pool_rounds,
                "gradient_threshold": args.gradient_threshold,
                "gradient_delta": args.gradient_delta,
                "max_add_per_round": args.max_add_per_round,
                "maxiter_per_optimization_stage": args.maxiter,
                "optimizer": "COBYLA",
                "optimizer_tolerance": args.optimizer_tolerance,
                "rhobeg": args.rhobeg,
                "construction_mode": args.construction_mode,
                "execution_mode": args.execution_mode,
                "candidate_initialization": args.candidate_initialization,
                "require_native_pauli": args.require_native_pauli,
                "trotter_order": args.trotter_order,
                "trotter_steps": args.trotter_steps,
            },
        },
        "molecules": {},
        "records": [],
        "summary": {},
    }
    write_json(output, payload)
    print(f"Output JSON: {output}")

    for name in args.molecules:
        print("=" * 96)
        print(f"Preparing molecule: {name}")
        molecule_payload: dict[str, Any] = {
            "status": "running",
            "problem": None,
            "engine": None,
            "workflow": None,
        }
        payload["molecules"][name] = molecule_payload
        write_json(output, payload)
        try:
            molecule, problem = prepare_molecule(name, args)
            molecule_payload["problem"] = problem
            if "engine" in args.tracks:
                molecule_payload["engine"] = run_engine_track(molecule, name, args)
                write_json(output, payload)
            if "workflow" in args.tracks:
                molecule_payload["workflow"] = run_workflow_track(molecule, name, args)
                write_json(output, payload)
            molecule_payload["status"] = "ok"
        except Exception as exc:
            molecule_payload.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(molecule_payload["traceback"], file=sys.stderr)
            if args.fail_fast:
                payload["status"] = "failed"
                payload["completed_at_utc"] = utc_now()
                write_json(output, payload)
                raise
        finally:
            payload["records"] = flat_records(payload)
            payload["summary"] = summarize(payload)
            write_json(output, payload)

    payload["status"] = (
        "ok"
        if all(item.get("status") == "ok" for item in payload["molecules"].values())
        else "completed_with_failures"
    )
    payload["completed_at_utc"] = utc_now()
    payload["records"] = flat_records(payload)
    payload["summary"] = summarize(payload)
    write_json(output, payload)
    print("=" * 96)
    print(f"Benchmark status: {payload['status']}")
    print(f"Final JSON: {output}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
