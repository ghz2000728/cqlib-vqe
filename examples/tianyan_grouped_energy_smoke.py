# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Submit one QWC-grouped energy evaluation to a Tianyan backend.

Required environment variable:
    TIANYAN_DEVICE       backend name, as returned by `tools/check_tianyan_api.py --online`

Optional environment variables:
    TIANYAN_API_KEY      omitted to reuse saved credentials
    TIANYAN_PHYSICAL_QUBIT physical qubit used for logical q0 (default: 0)
    TIANYAN_SHOTS        shots per measurement group (default: 2000)
    TIANYAN_CALIBRATION  enabled/disabled (default: disabled)
"""

from __future__ import annotations

import os

from cqlib import Circuit

from cqlib_vqe import TianyanEnergyEstimator


def main() -> None:
    device_name = os.environ["TIANYAN_DEVICE"]
    api_key = os.getenv("TIANYAN_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("TIANYAN_API_KEY is empty")
    physical_qubit = int(os.getenv("TIANYAN_PHYSICAL_QUBIT", "0"))
    shots = int(os.getenv("TIANYAN_SHOTS", "2000"))
    calibration_mode = os.getenv("TIANYAN_CALIBRATION", "disabled")

    # |+> has <X>=1 and <Z>=0.  X and Z are not QWC-compatible, so the
    # estimator creates two measurement circuits and submits them as one batch.
    circuit = Circuit(1)
    circuit.h(0)
    hamiltonian = [("I", 0.25), ("X", 1.0), ("Z", 0.5)]

    estimator = TianyanEnergyEstimator.from_credentials(
        device_name=device_name,
        n_qubits=1,
        api_key=api_key,
        shots=shots,
        calibration_mode=calibration_mode,
        physical_qubits=[physical_qubit],
        timeout_secs=600.0,
        poll_interval_secs=5.0,
    )

    result = estimator.evaluate_with_details(circuit, hamiltonian)
    print(f"device             : {result.device_name}")
    print(f"shots/group        : {result.shots}")
    print(f"measurement groups : {len(result.groups)}")
    print(f"task ids           : {list(result.task_ids)}")
    print(f"energy             : {result.energy:.10f}")
    for group in result.groups:
        print(
            f"  {group.master_pauli} on {group.measured_physical_qubits}: "
            f"contribution={group.contribution:+.10f}"
        )


if __name__ == "__main__":
    main()
