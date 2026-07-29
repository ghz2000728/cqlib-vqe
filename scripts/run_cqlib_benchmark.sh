#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRESET="${PRESET:-quick}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-$ROOT_DIR/run_outputs/cqlib_benchmark_${STAMP}/cqlib_results.json}"

if [[ -n "${MOLECULES:-}" ]]; then
  read -r -a MOLECULE_LIST <<< "${MOLECULES//,/ }"
else
  MOLECULE_LIST=(h2 h4 lih beh2 h2o)
fi

COMMON=(
  "${MOLECULE_LIST[@]}"
  --output "$OUTPUT"
  --active-space-policy "${ACTIVE_SPACE_POLICY:-default}"
  --initial-threshold "${INITIAL_THRESHOLD:-1e-3}"
  --pool-rounds "${POOL_ROUNDS:-1}"
  --gradient-threshold "${GRADIENT_THRESHOLD:-1e-3}"
  --gradient-delta "${GRADIENT_DELTA:-1e-3}"
  --max-add-per-round "${MAX_ADD_PER_ROUND:-2}"
  --maxiter "${MAXITER:-80}"
  --rhobeg "${RHOBEG:-0.03}"
  --construction-mode "${CONSTRUCTION_MODE:-bind}"
  --execution-mode "${EXECUTION_MODE:-auto}"
  --candidate-initialization "${CANDIDATE_INITIALIZATION:-best_probe}"
  --trotter-order "${TROTTER_ORDER:-2}"
  --trotter-steps "${TROTTER_STEPS:-2}"
  --seed "${SEED:-2026}"
)

if [[ "${REQUIRE_NATIVE_PAULI:-1}" == "0" ]]; then
  COMMON+=(--no-require-native-pauli)
else
  COMMON+=(--require-native-pauli)
fi

case "$PRESET" in
  smoke)
    EXTRA=(
      --tracks engine,workflow
      --engine-counts 10
      --engine-repeats 1
      --engine-warmup 2
      --workflow-repeats 1
      --variants static_selected,adaptive_r1
    )
    ;;
  quick)
    EXTRA=(
      --tracks engine,workflow
      --engine-counts 10,100
      --engine-repeats 3
      --engine-warmup 5
      --workflow-repeats 1
      --variants static_selected,adaptive_r1
    )
    ;;
  report)
    EXTRA=(
      --tracks engine,workflow
      --engine-counts 100
      --engine-repeats 5
      --engine-warmup 10
      --workflow-repeats 3
      --variants full_uccsd,static_selected,adaptive_r1
    )
    ;;
  engine)
    EXTRA=(
      --tracks engine
      --engine-counts "${ENGINE_COUNTS:-100}"
      --engine-repeats "${ENGINE_REPEATS:-5}"
      --engine-warmup "${ENGINE_WARMUP:-10}"
    )
    ;;
  workflow)
    EXTRA=(
      --tracks workflow
      --workflow-repeats "${WORKFLOW_REPEATS:-3}"
      --variants "${VARIANTS:-full_uccsd,static_selected,adaptive_r1}"
    )
    ;;
  *)
    echo "Unknown PRESET=$PRESET (expected smoke, quick, report, engine, workflow)" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUTPUT")"
echo "PRESET=$PRESET"
echo "OUTPUT=$OUTPUT"
echo "MOLECULES=${MOLECULE_LIST[*]}"
python -u benchmarks/run_cqlib_benchmark.py "${COMMON[@]}" "${EXTRA[@]}" "$@" \
  2>&1 | tee "${OUTPUT%.json}.log"
python benchmarks/validate_results.py "$OUTPUT"
python benchmarks/print_summary.py "$OUTPUT"
