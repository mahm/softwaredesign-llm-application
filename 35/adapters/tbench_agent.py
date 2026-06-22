import base64
import os
import shlex
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

from .agent_workdir import WORKDIR_BOOTSTRAP, WORKDIR_FILE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARNESS_FILE = PROJECT_ROOT / "harness-runs" / "baseline" / "harness.json"
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "container-setup.sh"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _bundle_source(harness_file: str | None) -> str:
    files = [
        "package.json",
        "bun.lock",
        "tsconfig.json",
    ]
    directories = [
        "src",
    ]

    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in files:
            source = PROJECT_ROOT / relative
            if source.exists():
                archive.add(source, arcname=relative)

        for relative in directories:
            source = PROJECT_ROOT / relative
            if source.exists():
                archive.add(source, arcname=relative)

        harness_path = Path(harness_file).expanduser() if harness_file else DEFAULT_HARNESS_FILE
        if not harness_path.is_absolute():
            harness_path = (PROJECT_ROOT / harness_path).resolve()
        archive.add(harness_path, arcname="harness.json")

    return base64.b64encode(buffer.getvalue()).decode("ascii")


class DeepAgentsTsAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "deepagents-ts-openrouter"

    def __init__(self, harness_file: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _load_env_file(PROJECT_ROOT / ".env")
        self._harness_file = harness_file

    @property
    def _env(self) -> dict[str, str]:
        env = {
            "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
        }

        for key in [
            "BASE_MODEL",
            "AGENT_WORKDIR",
            "OPENROUTER_PROVIDER_CONFIG",
            "DEEPAGENTS_TBENCH_REPO",
            "DEEPAGENTS_TBENCH_REF",
        ]:
            if os.environ.get(key):
                env[key] = os.environ[key]

        return env

    @property
    def _install_agent_script_path(self) -> Path:
        if os.environ.get("DEEPAGENTS_TBENCH_REPO"):
            return SETUP_SCRIPT

        bundle = _bundle_source(self._harness_file)
        generated = tempfile.NamedTemporaryFile(
            mode="w", suffix="-deepagents-ts-setup.sh", delete=False
        )
        generated.write("#!/usr/bin/env bash\n")
        generated.write("set -euo pipefail\n")
        generated.write(f"export DEEPAGENTS_TBENCH_BUNDLE_BASE64={shlex.quote(bundle)}\n")
        generated.write(SETUP_SCRIPT.read_text(encoding="utf-8"))
        generated.close()
        os.chmod(generated.name, 0o755)
        return Path(generated.name)

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        encoded_instruction = base64.b64encode(instruction.encode("utf-8")).decode(
            "ascii"
        )
        command = f"""
set -euo pipefail
printf '%s' {shlex.quote(encoded_instruction)} | base64 -d > /tmp/task.md
{WORKDIR_BOOTSTRAP}
agent_workdir="$(choose_agent_workdir /tmp/task.md)"
normalize_empty_app_alias "$agent_workdir"
printf '%s\n' "$agent_workdir" > {shlex.quote(WORKDIR_FILE)}
cd /opt/deepagents-tbench-autotune-ts
export AGENT_WORKDIR="$(cat {shlex.quote(WORKDIR_FILE)} 2>/dev/null || printf '%s' /app)"
run_status=0
harness_timeout="$(bun -e 'const h = await Bun.file("harness.json").json(); if (h.agentRunTimeoutSec) console.log(h.agentRunTimeoutSec)' 2>/dev/null || true)"
agent_timeout="${{AGENT_RUN_TIMEOUT_SEC:-${{harness_timeout:-840}}}}"
if command -v timeout >/dev/null 2>&1; then
  timeout "$agent_timeout" bun src/main.ts run /tmp/task.md /agent-logs/deepagents-ts harness.json || run_status=$?
else
  bun src/main.ts run /tmp/task.md /agent-logs/deepagents-ts harness.json || run_status=$?
fi
bun src/main.ts report /agent-logs/deepagents-ts || true
exit 0
"""

        return [
            TerminalCommand(
                command=command,
                max_timeout_sec=float("inf"),
                block=True,
            )
        ]
