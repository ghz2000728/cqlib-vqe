#!/usr/bin/env python3
"""Print a compact summary from a cqlib-vqe 1.4 benchmark JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENGINE_ORDER = (
    "native_pauli_objective",
    "decomposed_pauli_objective",
    "circuit_bind_objective",
    "circuit_jit_objective",
    "assign_parameters_only",
)


def fmt(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    print(
        f"framework={payload['framework']} status={payload['status']} "
        f"profile={payload.get('benchmark_profile', '-')}"
    )

    print("\nWORKFLOW")
    print("molecule  variant          time_s    error_mHa   params   evals  chem")
    print("--------  ---------------  --------  ----------  -------  -----  ----")
    for molecule_name, molecule in payload.get("molecules", {}).items():
        workflow = molecule.get("workflow") or {}
        for variant, data in workflow.get("variants", {}).items():
            for run in data.get("runs", []):
                if run.get("status") != "ok":
                    print(f"{molecule_name:8s}  {variant:15s}  FAILED: {run.get('error')}")
                    continue
                print(
                    f"{molecule_name:8s}  {variant:15s}  "
                    f"{fmt(run.get('wall_time_seconds'), 3):>8s}  "
                    f"{fmt(run.get('absolute_fci_error_mha'), 6):>10s}  "
                    f"{str(run.get('selected_parameter_count')):>7s}  "
                    f"{str(run.get('total_energy_evaluations')):>5s}  "
                    f"{str(bool(run.get('chemical_accuracy'))):>4s}"
                )

    print("\nENGINE (largest batch)")
    print("molecule  mode                           eval/s       us/eval")
    print("--------  -----------------------------  -----------  -----------")
    for molecule_name, molecule in payload.get("molecules", {}).items():
        engine = molecule.get("engine") or {}
        batches = engine.get("batches", {})
        if not batches:
            continue
        largest = max(batches, key=lambda value: int(value))
        batch = batches[largest]
        for mode in ENGINE_ORDER:
            item = batch.get(mode)
            if not item:
                continue
            rate = item["evaluations_per_second"]["median"]
            seconds = item["seconds_per_evaluation"]["median"]
            print(
                f"{molecule_name:8s}  {mode:29s}  "
                f"{fmt(rate, 3):>11s}  {fmt(seconds * 1e6, 3):>11s}"
            )
        speedups = batch.get("speedups", {})
        native_decomp = speedups.get("native_vs_decomposed")
        if native_decomp is not None:
            print(
                f"{'':8s}  {'native / decomposed speedup':29s}  "
                f"{native_decomp:>10.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
