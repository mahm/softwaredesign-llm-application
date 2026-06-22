#!/usr/bin/env bash
set -euo pipefail

cat > /app/service-config.json <<'JSON'
{
  "service": "demo-api",
  "enabled": true,
  "retries": 3,
  "endpoints": {
    "health": "/healthz"
  }
}
JSON
