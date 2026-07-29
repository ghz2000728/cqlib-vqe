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

ACTIVE_SPACE_POLICY="${ACTIVE_SPACE_POLICY:-default}"
INITIAL_THRESHOLD="${INITIAL_THRESHOLD:-1e-3}"
POOL_ROUNDS="${POOL_ROUNDS:-2}"
GRADIENT_THRESHOLD="${GRADIENT_THRESHOLD:-1e-4}"
MAXITER="${MAXITER:-80}"
TROTTER_STEPS="${TROTTER_STEPS:-2}"
CANDIDATE_INITIALIZATION="${CANDIDATE_INITIALIZATION:-best_probe}"

ARGS=(
  --active-space-policy "$ACTIVE_SPACE_POLICY"
  --initial-threshold "$INITIAL_THRESHOLD"
  --pool-rounds "$POOL_ROUNDS"
  --maxiter "$MAXITER"
  --rhobeg 0.03
  --gradient-delta 1e-3
  --gradient-threshold "$GRADIENT_THRESHOLD"
  --trotter-order 2
  --trotter-steps "$TROTTER_STEPS"
  --candidate-initialization "$CANDIDATE_INITIALIZATION"
  --construction-mode bind
  --execution-mode auto
)

# Empty means no cap: every generator whose |gradient| is strictly greater
# than GRADIENT_THRESHOLD is added in that pool scan.
if [[ -n "${MAX_ADD_PER_ROUND:-}" ]]; then
  ARGS+=(--max-add-per-round "$MAX_ADD_PER_ROUND")
fi

python -u examples/run_five_adaptive_full_pool.py "${ARGS[@]}" "$@"
