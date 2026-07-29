#!/usr/bin/env bash
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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="run_outputs/accuracy_${STAMP}"
mkdir -p "$OUT"

run_one() {
  local mol="$1"
  local maxiter="$2"
  local poolrounds="$3"
  echo "========== ${mol} ==========" | tee "$OUT/${mol}.log"
  python -u examples/run_local_accuracy.py "$mol" \
    --pool-mode adaptive \
    --initial-threshold 1e-3 \
    --candidate-threshold 0.0 \
    --pool-rounds "$poolrounds" \
    --max-add-per-round 2 \
    --maxiter "$maxiter" \
    --rhobeg 0.03 \
    --tol 1e-7 \
    --gradient-delta 1e-3 \
    --gradient-threshold 1e-4 \
    --trotter-order 2 \
    --trotter-steps 2 \
    --construction-mode bind \
    --execution-mode auto \
    --require-chemical-accuracy \
    2>&1 | tee -a "$OUT/${mol}.log"
}

run_one h2   80 2
run_one h4   80 2
run_one lih  80 2
run_one beh2 80 2
run_one h2o  80 2

echo "All five systems reached the requested chemical-accuracy check."
echo "Logs: $OUT"
