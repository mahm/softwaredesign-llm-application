#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is required in 35/.env or the environment" >&2
  exit 1
fi

mkdir -p results/tbench tmp

RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"
HARNESS_FILE="${HARNESS_FILE:-harness-runs/baseline/harness.json}"

uvx --from terminal-bench tb run \
  --dataset-path smoke_tasks \
  --agent-import-path adapters.tbench_agent:DeepAgentsTsAgent \
  --agent-kwarg "harness_file=$HARNESS_FILE" \
  --output-path results/tbench \
  --run-id "$RUN_ID" \
  --task-id create-answer-file \
  --task-id repair-json-config \
  --n-concurrent 1 \
  --global-agent-timeout-sec 300 \
  --global-test-timeout-sec 120 \
  --no-cleanup

cat > results/tbench/latest-smoke-runs.txt <<EOF
run=$RUN_ID
harness=$HARNESS_FILE
EOF

echo "Terminal-Bench smoke run complete:"
cat results/tbench/latest-smoke-runs.txt
