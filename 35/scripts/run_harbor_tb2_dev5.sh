#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <label> <openrouter-model-id>" >&2
  exit 2
fi

label="$1"
model="$2"
job="tb2-${label}-dev5-$(date +%Y%m%d-%H%M%S)"
harness_file="${HARNESS_FILE:-harness-runs/baseline/harness.json}"
attempts="${N_ATTEMPTS:-3}"
use_shared_bridge="${USE_SHARED_DOCKER_BRIDGE:-0}"

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

tasks=(
  terminal-bench/prove-plus-comm
  terminal-bench/fix-git
  terminal-bench/openssl-selfsigned-cert
  terminal-bench/log-summary-date-ranges
  terminal-bench/regex-log
)

concurrency="${CONCURRENCY:-6}"
if [ "$concurrency" = "all" ]; then
  concurrency="$(( ${#tasks[@]} * attempts ))"
fi

docker_config="${DOCKER_CONFIG:-$PWD/tmp/docker-clean-config}"
mkdir -p "$docker_config/cli-plugins"
if [ -x "$HOME/.docker/cli-plugins/docker-compose" ]; then
  ln -sf "$HOME/.docker/cli-plugins/docker-compose" "$docker_config/cli-plugins/docker-compose"
fi
if [ -x "$HOME/.docker/cli-plugins/docker-buildx" ]; then
  ln -sf "$HOME/.docker/cli-plugins/docker-buildx" "$docker_config/cli-plugins/docker-buildx"
fi

task_args=()
for task in "${tasks[@]}"; do
  task_args+=("-i" "$task")
done

harbor_args=(
  run
  -d terminal-bench/terminal-bench-2
  --agent-import-path adapters.harbor_agent:DeepAgentsTsHarborAgent
  --agent-kwarg "harness_file=${harness_file}"
  --agent-kwarg "trace_mode=full"
  --agent-env "BASE_MODEL=${model}"
  "${task_args[@]}"
)
for key in AGENT_WORKDIR AGENT_RUN_TIMEOUT_SEC OPENROUTER_PROVIDER_CONFIG DEEPAGENTS_TBENCH_REPO DEEPAGENTS_TBENCH_REF; do
  value="${!key-}"
  if [ -n "$value" ]; then
    harbor_args+=(--agent-env "${key}=${value}")
  fi
done
if [ "$use_shared_bridge" = "1" ]; then
  harbor_args+=("--extra-docker-compose" "docker/docker-compose.shared-bridge.yaml")
fi
harbor_args+=(
  -k "$attempts"
  -n "$concurrency"
  -o results/harbor
  --job-name "$job"
  --yes
  --debug
)

DOCKER_CONFIG="$docker_config" harbor "${harbor_args[@]}"

mkdir -p results/harbor
bun src/harbor-report.ts "results/harbor/${job}" "$attempts"
printf '%s=%s\n' "$label" "$job" | tee -a results/harbor/latest-tb2-dev5.txt
