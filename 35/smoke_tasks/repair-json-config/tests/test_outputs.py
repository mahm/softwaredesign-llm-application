import json
from pathlib import Path


def test_service_config_values():
    config = json.loads(Path("/app/service-config.json").read_text(encoding="utf-8"))

    assert config == {
        "service": "demo-api",
        "enabled": True,
        "retries": 3,
        "endpoints": {
            "health": "/healthz",
        },
    }
