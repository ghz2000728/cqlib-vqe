#!/usr/bin/env python3
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

"""Validate the cqlib-vqe 1.4 benchmark JSON contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SCHEMA = "cqlib-vqe-benchmark/1.1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "framework",
        "status",
        "environment",
        "configuration",
        "molecules",
        "records",
        "summary",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing top-level fields: {missing}")
    if payload["schema_version"] != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected schema version: {payload['schema_version']!r}")
    if payload["framework"] != "cqlib":
        raise ValueError(f"Unexpected framework: {payload['framework']!r}")
    if not isinstance(payload["records"], list):
        raise TypeError("records must be a list")

    engine_requested = "engine" in payload["configuration"].get("tracks", [])
    if engine_requested:
        for name, molecule in payload.get("molecules", {}).items():
            if molecule.get("status") != "ok":
                continue
            engine = molecule.get("engine") or {}
            if not engine.get("native_pauli_rotation_available"):
                raise ValueError(f"{name}: native PauliRotation was not available")
            for count, batch in engine.get("batches", {}).items():
                if "native_pauli_objective" not in batch:
                    raise ValueError(f"{name}/{count}: missing native_pauli_objective")
                delta = engine.get("energy_consistency", {}).get("max_absolute_delta_ha")
                if delta is None or float(delta) > 1e-9:
                    raise ValueError(f"{name}: engine energy mismatch {delta}")

    print(f"schema: {payload['schema_version']}")
    print(f"framework: {payload['framework']}")
    print(f"status: {payload['status']}")
    print(f"molecules: {len(payload['molecules'])}")
    print(f"flat records: {len(payload['records'])}")
    print("cqlib-vqe 1.4 benchmark JSON validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
