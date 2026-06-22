#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y python3-pytest

cd /app
pytest "$TEST_DIR/test_outputs.py" -rA
