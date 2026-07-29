"""Validate the cqlib-tianyan API and the VQE grouped-measurement adapter."""

from __future__ import annotations

import argparse
import os

from cqlib import Circuit
from cqlib_tianyan import TianyanBackend, TianyanPlatform, TaskHandle

from cqlib_vqe import TianyanEnergyEstimator


def check_types() -> None:
    required_platform = ("login", "from_credentials", "list_backends", "get_backend", "submit")
    required_backend = ("run", "run_raw", "run_with_mode", "device_config")
    required_task = ("status", "wait", "wait_raw")
    for owner, names in (
        (TianyanPlatform, required_platform),
        (TianyanBackend, required_backend),
        (TaskHandle, required_task),
    ):
        missing = [name for name in names if not hasattr(owner, name)]
        if missing:
            raise RuntimeError(f"{owner.__name__} is missing methods: {missing}")


class _DryRunBackend:
    name = "dry-run"
    num_qubits = 16

    def run_with_mode(self, circuits, shots, mode="auto"):
        raise AssertionError("dry-run validation must not submit hardware jobs")


def check_grouping() -> None:
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    estimator = TianyanEnergyEstimator(
        _DryRunBackend(),
        2,
        shots=1000,
        physical_qubits=[1, 8],
    )
    plan = estimator.prepare(
        circuit,
        [("II", 0.5), ("XI", 1.0), ("IX", 2.0), ("XX", -0.25), ("ZZ", 0.75)],
    )
    if plan.constant_energy != 0.5 or plan.num_measurement_circuits != 2:
        raise RuntimeError(f"unexpected grouping plan: {plan}")
    if not all("M Q" in script for script in plan.circuits):
        raise RuntimeError("measurement QCIS was not appended")


def online_check(device_name: str | None) -> None:
    api_key = os.getenv("TIANYAN_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("TIANYAN_API_KEY is empty")
    platform = (
        TianyanPlatform.login(api_key)
        if api_key
        else TianyanPlatform.from_credentials()
    )
    backends = platform.list_backends()
    print(f"online backends: {len(backends)}")
    if device_name:
        backend = platform.get_backend(device_name)
        print(
            f"selected backend: {backend.name}, status={backend.status}, "
            f"qubits={backend.num_qubits}"
        )
        device = backend.device_config()
        print(
            f"device config: usable={device.num_usable_qubits}, "
            f"couplings={device.topology.num_couplings}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="authenticate and list backends")
    parser.add_argument("--device", help="backend to inspect during --online check")
    args = parser.parse_args()

    check_types()
    check_grouping()
    print("cqlib-tianyan type surface: OK")
    print("VQE QWC/QCIS dry-run: OK")
    if args.online:
        online_check(args.device)


if __name__ == "__main__":
    main()
