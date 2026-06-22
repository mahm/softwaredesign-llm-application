#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <harbor-job-dir>" >&2
  exit 2
fi

job_dir="$1"
if [ ! -d "$job_dir" ]; then
  echo "Harbor job directory not found: $job_dir" >&2
  exit 1
fi

find "$job_dir" -mindepth 1 -maxdepth 1 \
  ! -name result.json \
  ! -name suite-summary.json \
  -exec rm -rf {} +
