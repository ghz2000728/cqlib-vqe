#!/usr/bin/env python3
"""Run H2, H4, LiH, BeH2, and H2O with the 1.4 native-Pauli adaptive method.

Correctness rule enforced by this runner:

* CCSD amplitudes select only the initial ansatz.
* The adaptive candidate pool is the complete canonical singlet-UCCSD
  complement of the current active ansatz.
* Exact-zero CCSD-amplitude generators remain candidates and are ranked by
  their variational energy gradients.

The runner executes every requested molecule even if one misses chemical
accuracy.  It writes one log and one JSON result per molecule plus a CSV
summary for the whole suite.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import math
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np

# Force imports from the checked-out project, not a stale wheel in site-packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cqlib import Circuit  # noqa: E402
from cqlib_vqe import ActiveSpaceConfig, MolecularDataEngine  # noqa: E402
from cqlib_vqe import (  # noqa: E402
    AdaptiveSelectedUCCSDSolver,
    AdaptiveSelectionConfig,
    DirectStatevectorEstimator,
    NativeStatevectorEstimator,
    UCCSDFactory,
)

CHEMICAL_ACCURACY_HA = 1.6e-3

MOLECULES: dict[str, dict[str, Any]] = {
    "h2": {
        "geometry": [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.735)),
        ],
        "default_active_qubits": None,
        "maxiter": 80,
        "pool_rounds": 2,
    },
    "h4": {
        "geometry": [
            ("H", (0.0, 0.0, -1.5)),
            ("H", (0.0, 0.0, -0.5)),
            ("H", (0.0, 0.0, 0.5)),
            ("H", (0.0, 0.0, 1.5)),
        ],
        "default_active_qubits": None,
        "maxiter": 80,
        "pool_rounds": 2,
    },
    "lih": {
        "geometry": [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.596)),
        ],
        "default_active_qubits": 10,
        "maxiter": 80,
        "pool_rounds": 2,
    },
    "beh2": {
        "geometry": [
            ("H", (0.0, 0.0, -1.3264)),
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.3264)),
        ],
        "default_active_qubits": 12,
        "maxiter": 80,
        "pool_rounds": 2,
    },
    "h2o": {
        "geometry": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.587)),
            ("H", (0.0, -0.757, 0.587)),
        ],
        "default_active_qubits": 12,
        "maxiter": 80,
        "pool_rounds": 2,
    },
}


class Tee:
    """Write text to the terminal and a persistent log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _active_space(case: dict[str, Any], policy: str) -> ActiveSpaceConfig:
    if policy == "full":
        return ActiveSpaceConfig.full()
    budget = case["default_active_qubits"]
    if budget is None:
        return ActiveSpaceConfig.full()
    return ActiveSpaceConfig.automatic(
        max_active_qubits=int(budget),
        freeze_core=True,
    )


def _build_molecule(name: str, args: argparse.Namespace):
    case = MOLECULES[name]
    return MolecularDataEngine(
        geometry=case["geometry"],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        mapper_type="jw",
        excitation_threshold=args.initial_threshold,
        active_space=_active_space(case, args.active_space_policy),
    ).run(run_fci=True, audit_ccsd_basis=True)


def _check_mapped_hf(molecule) -> float:
    circuit = Circuit(molecule.n_qubits)
    for qubit in range(molecule.n_electrons):
        circuit.x(qubit)
    estimator = NativeStatevectorEstimator(n_qubits=molecule.n_qubits)
    mapped_hf = float(estimator.evaluate(circuit, molecule.hamiltonian_data))
    delta = mapped_hf - float(molecule.hf_energy)
    print(f"mapped HF - PySCF HF = {delta:+.3e} Ha")
    if abs(delta) > 1e-8:
        raise AssertionError(
            "Active-space Hamiltonian/reference-state consistency failed: "
            f"delta={delta:+.6e} Ha"
        )
    return delta


def _amplitude_map(molecule) -> dict[int, float]:
    descriptors = list(molecule.ccsd_descriptors)
    amplitudes = np.asarray(molecule.ccsd_parameters, dtype=float)
    if len(descriptors) != amplitudes.size:
        raise ValueError("Descriptor/amplitude count mismatch")
    return {
        int(descriptor.packed_index): float(amplitude)
        for descriptor, amplitude in zip(descriptors, amplitudes, strict=True)
    }


def _initial_indices(amplitudes: dict[int, float], threshold: float) -> list[int]:
    selected = [
        index for index, value in amplitudes.items() if abs(value) >= threshold
    ]
    if not selected:
        selected = [max(amplitudes, key=lambda index: abs(amplitudes[index]))]
    return sorted(selected)


def _initial_ansatz_energy(
    molecule,
    estimator: DirectStatevectorEstimator,
    indices: list[int],
    amplitudes: dict[int, float],
    args: argparse.Namespace,
) -> float:
    factory = UCCSDFactory(
        molecule,
        construction_mode=args.construction_mode,
        trotter_steps=args.trotter_steps,
        trotter_order=args.trotter_order,
        selected_packed_indices=indices,
    )
    values = np.asarray([amplitudes[index] for index in indices], dtype=float)
    return float(
        estimator.evaluate_parameters(factory, values, molecule.hamiltonian_data)
    )


def _run_one(name: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    case = MOLECULES[name]
    log_path = out_dir / f"{name}.log"
    json_path = out_dir / f"{name}.json"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.__stdout__, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("=" * 88)
            print(f"molecule                 = {name}")
            print(f"project root             = {PROJECT_ROOT}")
            import vqe

            print(f"vqe source               = {Path(vqe.__file__).resolve()}")
            print(f"installed package version= {importlib.metadata.version('cqlib-vqe')}")
            print(f"active-space policy      = {args.active_space_policy}")
            print(f"initial amplitude cutoff = {args.initial_threshold:.3e}")
            print("candidate amplitude cut  = NONE (complete canonical complement)")
            print(f"operator-pool scans      = {args.pool_rounds}")
            print(f"gradient add threshold   = {args.gradient_threshold:.3e} (strict >)")
            print(f"max add per scan         = {args.max_add_per_round}")
            print(f"new parameter init       = {args.candidate_initialization}")
            print("=" * 88)

            molecule = _build_molecule(name, args)
            hf_mapping_error = _check_mapped_hf(molecule)
            estimator = DirectStatevectorEstimator(n_qubits=molecule.n_qubits)
            amplitudes = _amplitude_map(molecule)
            full_pool = tuple(range(int(molecule.ccsd_full_parameter_count)))
            initial = _initial_indices(amplitudes, args.initial_threshold)
            candidates = tuple(index for index in full_pool if index not in set(initial))
            zero_pool = tuple(
                index for index in full_pool if abs(amplitudes[index]) <= args.zero_tol
            )
            zero_candidates = tuple(index for index in candidates if index in set(zero_pool))

            print("\n=== pool partition ===")
            print(f"full pool count          = {len(full_pool)}")
            print(f"initial selected count   = {len(initial)}")
            print(f"candidate count          = {len(candidates)}")
            print(f"exact/near-zero pool     = {list(zero_pool)}")
            print(f"zero-amplitude candidates= {list(zero_candidates)}")

            initial_energy = _initial_ansatz_energy(
                molecule, estimator, initial, amplitudes, args
            )
            fci = float(molecule.fci_energy)
            print("\n=== initial CCSD-seeded factorized ansatz ===")
            print(f"energy                   = {initial_energy:+.12f} Ha")
            print(f"FCI error                = {(initial_energy - fci) * 1000:+.6f} mHa")

            config = AdaptiveSelectionConfig(
                initial_amplitude_threshold=args.initial_threshold,
                candidate_amplitude_threshold=0.0,
                pool_rounds=(
                    args.pool_rounds
                    if args.pool_rounds is not None
                    else int(case["pool_rounds"])
                ),
                max_add_per_round=args.max_add_per_round,
                max_parameters=args.max_parameters,
                max_screen_candidates=None,
                gradient_delta=args.gradient_delta,
                gradient_threshold=args.gradient_threshold,
                target_energy_error=args.target_error,
                optimizer_maxiter=args.maxiter or int(case["maxiter"]),
                optimizer_tolerance=args.tol,
                optimizer_options={"rhobeg": args.rhobeg, "catol": 1e-8},
                construction_mode=args.construction_mode,
                execution_mode=args.execution_mode,
                trotter_steps=args.trotter_steps,
                trotter_order=args.trotter_order,
                candidate_initialization=args.candidate_initialization,
            )
            solver = AdaptiveSelectedUCCSDSolver(molecule, estimator, config)

            round_zero_audit: list[dict[str, Any]] = []
            round_best = math.inf

            def optimizer_callback(_parameters, energy: float, evaluation: int) -> None:
                nonlocal round_best
                round_best = min(round_best, float(energy))
                if evaluation == 1 or evaluation % args.optimizer_log_every == 0:
                    print(
                        f"  optimizer eval={evaluation:4d} "
                        f"current={energy:+.12f} round_best={round_best:+.12f}"
                    )

            def round_callback(item) -> None:
                nonlocal round_best
                zero_ranked = [
                    candidate
                    for candidate in item.candidates
                    if abs(candidate.ccsd_amplitude) <= args.zero_tol
                ]
                added_zero = [
                    index
                    for index in item.added_indices
                    if abs(amplitudes[index]) <= args.zero_tol
                ]
                strongest_zero = zero_ranked[0] if zero_ranked else None
                round_zero_audit.append(
                    {
                        "round": int(item.round_index),
                        "added_indices": list(item.added_indices),
                        "added_zero_amplitude_indices": added_zero,
                        "strongest_zero_amplitude_candidate": (
                            None if strongest_zero is None else asdict(strongest_zero)
                        ),
                    }
                )
                error_mha = (
                    None
                    if item.reference_error is None
                    else 1000.0 * float(item.reference_error)
                )
                print("\n--- adaptive round ---")
                print(f"optimization round       = {item.round_index}")
                print(f"pool scan index          = {item.pool_scan_index}")
                print(f"active before            = {list(item.selected_before)}")
                print(f"candidate count          = {len(item.candidate_pool_before)}")
                print(f"optimized energy         = {item.optimized_energy:+.12f} Ha")
                print(f"absolute FCI error       = {error_mha} mHa")
                print(f"gradient norm            = {item.gradient_norm}")
                print(f"eligible above threshold = {list(item.eligible_indices)}")
                print(f"added                    = {list(item.added_indices)}")
                print(f"added zero-amplitude     = {added_zero}")
                if item.candidates:
                    print(f"top {min(args.top_k, len(item.candidates))} gradients:")
                    for rank, candidate in enumerate(item.candidates[: args.top_k], 1):
                        marker = " ZERO_CCSD" if abs(candidate.ccsd_amplitude) <= args.zero_tol else ""
                        print(
                            f"  #{rank:02d} packed={candidate.packed_index:3d} "
                            f"grad={candidate.gradient:+.9e} "
                            f"|grad|={candidate.absolute_gradient:.9e} "
                            f"ccsd={candidate.ccsd_amplitude:+.9e} "
                            f"init={candidate.initial_value:+.3e}{marker}"
                        )
                    if strongest_zero is not None:
                        zero_rank = next(
                            index + 1
                            for index, candidate in enumerate(item.candidates)
                            if candidate.packed_index == strongest_zero.packed_index
                        )
                        print(
                            "strongest zero-amplitude candidate: "
                            f"rank={zero_rank}, packed={strongest_zero.packed_index}, "
                            f"gradient={strongest_zero.gradient:+.9e}"
                        )
                round_best = math.inf

            result = solver.run(
                reference_energy=fci,
                round_callback=round_callback,
                optimizer_callback=optimizer_callback,
            )
            result_dict = result.to_dict()
            final_error = float(result.optimal_value - fci)
            chemical_accuracy = abs(final_error) <= args.target_error
            added_zero_all = sorted(
                {
                    index
                    for record in round_zero_audit
                    for index in record["added_zero_amplitude_indices"]
                }
            )

            payload = {
                "molecule": name,
                "basis": "sto-3g",
                "active_space_policy": args.active_space_policy,
                "active_space": molecule.active_space_report.to_dict(),
                "n_qubits": int(molecule.n_qubits),
                "n_electrons": int(molecule.n_electrons),
                "references": dict(molecule.reference_energies),
                "mapped_hf_minus_pyscf_hf": hf_mapping_error,
                "initial_threshold": args.initial_threshold,
                "zero_tolerance": args.zero_tol,
                "full_pool_indices": list(full_pool),
                "initial_selected_indices": initial,
                "initial_candidate_indices": list(candidates),
                "zero_amplitude_pool_indices": list(zero_pool),
                "zero_amplitude_candidate_indices": list(zero_candidates),
                "initial_ccsd_seeded_energy": initial_energy,
                "initial_ccsd_seeded_fci_error_ha": initial_energy - fci,
                "adaptive_result": result_dict,
                "pool_rounds_requested": config.resolved_pool_rounds,
                "pool_scans_completed": result.pool_scans_completed,
                "gradient_threshold": config.gradient_threshold,
                "max_add_per_round": config.resolved_max_add_per_round,
                "round_zero_amplitude_audit": round_zero_audit,
                "added_zero_amplitude_indices": added_zero_all,
                "final_energy": float(result.optimal_value),
                "final_fci_error_ha": final_error,
                "final_fci_error_mha": 1000.0 * final_error,
                "chemical_accuracy": chemical_accuracy,
            }
            json_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print("\n=== final result ===")
            print(f"energy                   = {result.optimal_value:+.12f} Ha")
            print(f"FCI                      = {fci:+.12f} Ha")
            print(f"signed FCI error         = {1000.0 * final_error:+.6f} mHa")
            print(f"chemical accuracy        = {chemical_accuracy}")
            print(f"initial selected         = {list(result.initial_selected_packed_indices)}")
            print(f"final selected           = {list(result.selected_packed_indices)}")
            print(f"added zero-amplitude     = {added_zero_all}")
            print(f"remaining candidates     = {list(result.remaining_candidate_packed_indices)}")
            print(f"stop reason              = {result.stop_reason}")
            print(f"pool scans completed     = {result.pool_scans_completed}")
            print(f"total evaluations        = {result.total_evaluations}")
            print(f"JSON                     = {json_path}")

            return {
                "molecule": name,
                "status": "ok",
                "active_space_policy": args.active_space_policy,
                "qubits": int(molecule.n_qubits),
                "electrons": int(molecule.n_electrons),
                "full_pool": len(full_pool),
                "initial_params": len(initial),
                "final_params": len(result.selected_packed_indices),
                "zero_candidates": len(zero_candidates),
                "added_zero_indices": " ".join(map(str, added_zero_all)),
                "initial_energy": initial_energy,
                "final_energy": float(result.optimal_value),
                "fci_energy": fci,
                "initial_error_mha": 1000.0 * (initial_energy - fci),
                "final_error_mha": 1000.0 * final_error,
                "chemical_accuracy": chemical_accuracy,
                "pool_scans": int(result.pool_scans_completed),
                "gradient_threshold": float(config.gradient_threshold),
                "max_add_per_round": (
                    ""
                    if config.resolved_max_add_per_round is None
                    else int(config.resolved_max_add_per_round)
                ),
                "evaluations": int(result.total_evaluations),
                "stop_reason": result.stop_reason,
                "log": str(log_path),
                "json": str(json_path),
            }


def _failed_summary(name: str, out_dir: Path, exc: BaseException) -> dict[str, Any]:
    return {
        "molecule": name,
        "status": "failed",
        "active_space_policy": "",
        "qubits": "",
        "electrons": "",
        "full_pool": "",
        "initial_params": "",
        "final_params": "",
        "zero_candidates": "",
        "added_zero_indices": "",
        "initial_energy": "",
        "final_energy": "",
        "fci_energy": "",
        "initial_error_mha": "",
        "final_error_mha": "",
        "chemical_accuracy": False,
        "pool_scans": "",
        "gradient_threshold": "",
        "max_add_per_round": "",
        "evaluations": "",
        "stop_reason": f"{type(exc).__name__}: {exc}",
        "log": str(out_dir / f"{name}.log"),
        "json": "",
    }



def _preflight_framework() -> None:
    """Fail immediately when the checked-out source is not the complete API set."""
    import chemistry
    import vqe

    project = PROJECT_ROOT.resolve()
    modules = {
        "vqe": Path(vqe.__file__).resolve(),
        "chemistry": Path(chemistry.__file__).resolve(),
    }
    for name, path in modules.items():
        try:
            path.relative_to(project)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} was imported from {path}, not from the checked-out project {project}. "
                "Install this project in editable mode and remove stale mixed-source installs."
            ) from exc

    factory_parameters = inspect.signature(UCCSDFactory.__init__).parameters
    required_factory_parameters = {
        "selected_packed_indices",
        "trotter_order",
        "trotter_steps",
    }
    missing = sorted(required_factory_parameters - set(factory_parameters))
    if missing:
        raise RuntimeError(
            "The local vqe/factory.py is older than the adaptive solver. "
            f"Missing UCCSDFactory parameters: {missing}. "
            "Apply the complete source-sync overlay; partial overlays are unsupported."
        )

    adaptive_parameters = inspect.signature(AdaptiveSelectionConfig).parameters
    required_adaptive_parameters = {
        "initial_amplitude_threshold",
        "candidate_amplitude_threshold",
        "pool_rounds",
        "gradient_threshold",
        "max_add_per_round",
        "max_screen_candidates",
    }
    missing = sorted(required_adaptive_parameters - set(adaptive_parameters))
    if missing:
        raise RuntimeError(
            "The local vqe/adaptive.py is incompatible with the five-molecule runner. "
            f"Missing AdaptiveSelectionConfig fields: {missing}."
        )

    print("framework preflight       = PASS")
    print(f"vqe source               = {modules['vqe']}")
    print(f"chemistry source         = {modules['chemistry']}")
    print(f"factory signature        = {inspect.signature(UCCSDFactory.__init__)}")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Five-molecule adaptive canonical full-pool UCCSD audit"
    )
    parser.add_argument(
        "molecules",
        nargs="*",
        metavar="MOLECULE",
        help="subset of: h2 h4 lih beh2 h2o; omitted means all five",
    )
    parser.add_argument(
        "--active-space-policy",
        choices=["default", "full"],
        default="default",
        help="default freezes core for LiH/BeH2/H2O; full retains every STO-3G orbital",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-threshold", type=float, default=1e-3)
    parser.add_argument("--zero-tol", type=float, default=1e-14)
    parser.add_argument(
        "--pool-rounds",
        "--max-rounds",
        dest="pool_rounds",
        type=int,
        default=2,
        help="maximum number of complete operator-pool gradient scans",
    )
    parser.add_argument(
        "--max-add-per-round",
        "--batch-size",
        dest="max_add_per_round",
        type=int,
        default=None,
        help="optional cap; default adds every candidate above the gradient threshold",
    )
    parser.add_argument("--max-parameters", type=int)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--rhobeg", type=float, default=0.03)
    parser.add_argument("--gradient-delta", type=float, default=1e-3)
    parser.add_argument("--gradient-threshold", type=float, default=1e-4)
    parser.add_argument("--target-error", type=float, default=CHEMICAL_ACCURACY_HA)
    parser.add_argument("--trotter-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--trotter-steps", type=int, default=2)
    parser.add_argument(
        "--candidate-initialization",
        choices=["best_probe", "zero"],
        default="best_probe",
        help="initialize added generators from the best already-evaluated ±delta probe",
    )
    parser.add_argument("--construction-mode", choices=["jit", "bind"], default="bind")
    parser.add_argument(
        "--execution-mode",
        choices=["auto", "circuit", "direct_statevector"],
        default="auto",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--optimizer-log-every", type=int, default=20)
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="return a nonzero exit code when any completed molecule misses chemical accuracy",
    )
    args = parser.parse_args()
    if not args.molecules:
        args.molecules = list(MOLECULES)

    _preflight_framework()

    invalid = [name for name in args.molecules if name not in MOLECULES]
    if invalid:
        parser.error(
            f"unknown molecule(s): {invalid}; choose from {list(MOLECULES)}"
        )
    if args.initial_threshold < 0 or args.zero_tol < 0:
        parser.error("thresholds must be non-negative")
    if args.pool_rounds < 0:
        parser.error("pool rounds must be non-negative")
    if args.max_add_per_round is not None and args.max_add_per_round <= 0:
        parser.error("max-add-per-round must be positive when supplied")
    if args.top_k <= 0 or args.optimizer_log_every <= 0:
        parser.error("top-k and log interval must be positive")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or PROJECT_ROOT / "run_outputs" / f"five_adaptive_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for name in args.molecules:
        try:
            summaries.append(_run_one(name, args, out_dir))
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # Intentionally continue so all five are tested.
            rendered = traceback.format_exc()
            print(rendered, file=sys.__stderr__, flush=True)
            with (out_dir / f"{name}.log").open("a", encoding="utf-8") as handle:
                handle.write("\n=== uncaught exception ===\n")
                handle.write(rendered)
            summaries.append(_failed_summary(name, out_dir, exc))

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_json.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fieldnames = list(summaries[0]) if summaries else []
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 88)
    print("five-molecule summary")
    print("=" * 88)
    for item in summaries:
        print(
            f"{item['molecule']:5s} status={item['status']:6s} "
            f"params={item['final_params']}/{item['full_pool']} "
            f"error_mHa={item['final_error_mha']} "
            f"chem_acc={item['chemical_accuracy']} "
            f"pool_scans={item['pool_scans']} "
            f"zero_added={item['added_zero_indices']}"
        )
    print(f"summary CSV  = {summary_csv}")
    print(f"summary JSON = {summary_json}")

    any_failed = any(item["status"] != "ok" for item in summaries)
    any_miss = any(
        item["status"] == "ok" and not bool(item["chemical_accuracy"])
        for item in summaries
    )
    return 1 if any_failed or (args.fail_on_miss and any_miss) else 0


if __name__ == "__main__":
    raise SystemExit(main())
