import base64
import os
import shlex
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .agent_workdir import WORKDIR_BOOTSTRAP, WORKDIR_FILE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARNESS_FILE = PROJECT_ROOT / "harness-runs" / "baseline" / "harness.json"
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "container-setup.sh"
APP_DIR = "/opt/deepagents-tbench-autotune-ts"
AGENT_LOG_DIR = "/logs/agent/deepagents-ts"
NO_TRACE_OUT_DIR = "/tmp/deepagents-ts-no-trace"
DEFAULT_OUTER_AGENT_TIMEOUT_SEC = 7200


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


class DeepAgentsTsHarborAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "deepagents-ts-openrouter"

    def __init__(
        self,
        harness_file: str | None = None,
        trace_mode: str = "full",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        _load_env_file(PROJECT_ROOT / ".env")
        self._harness_file = harness_file
        self._trace_mode = str(trace_mode or "full").lower()
        if self._trace_mode not in {"full", "none"}:
            raise ValueError("trace_mode must be 'full' or 'none'")

    def _agent_run_timeout_sec(self) -> int:
        configured = self._get_env("AGENT_RUN_TIMEOUT_SEC")

        try:
            return int(configured) if configured else DEFAULT_OUTER_AGENT_TIMEOUT_SEC
        except (TypeError, ValueError):
            return DEFAULT_OUTER_AGENT_TIMEOUT_SEC

    @property
    def _env(self) -> dict[str, str]:
        api_key = self._get_env("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required in 35/.env or environment")

        env = {
            "OPENROUTER_API_KEY": api_key,
        }

        for key in [
            "BASE_MODEL",
            "AGENT_WORKDIR",
            "AGENT_RUN_TIMEOUT_SEC",
            "OPENROUTER_PROVIDER_CONFIG",
            "DEEPAGENTS_TBENCH_REPO",
            "DEEPAGENTS_TBENCH_REF",
        ]:
            value = self._get_env(key)
            if value:
                env[key] = value

        return env

    async def install(self, environment: BaseEnvironment) -> None:
        bundle = _bundle_source(self._harness_file)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="-deepagents-ts-harbor-setup.sh", delete=False
        ) as generated:
            generated.write("#!/usr/bin/env bash\n")
            generated.write("set -euo pipefail\n")
            generated.write(f"export DEEPAGENTS_TBENCH_BUNDLE_BASE64={shlex.quote(bundle)}\n")
            generated.write(SETUP_SCRIPT.read_text(encoding="utf-8"))
            setup_path = Path(generated.name)

        os.chmod(setup_path, 0o755)
        await environment.upload_file(setup_path, "/tmp/deepagents-ts-harbor-setup.sh")
        await self.exec_as_root(
            environment,
            "bash /tmp/deepagents-ts-harbor-setup.sh",
            env=self._env,
            timeout_sec=900,
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        encoded_instruction = base64.b64encode(instruction.encode("utf-8")).decode(
            "ascii"
        )
        if self._trace_mode == "full":
            trace_prepare = f"""
mkdir -p {shlex.quote(AGENT_LOG_DIR)}
printf '%s\n' "$agent_workdir" > {shlex.quote(AGENT_LOG_DIR)}/workdir.txt
"""
            trace_run_setup = f"mkdir -p {shlex.quote(AGENT_LOG_DIR)}"
            run_invocation = (
                "bun src/main.ts run /tmp/task.md "
                f"{shlex.quote(AGENT_LOG_DIR)} harness.json full"
            )
            report_invocation = (
                f"bun src/main.ts report {shlex.quote(AGENT_LOG_DIR)} || true"
            )
        else:
            trace_prepare = ":"
            trace_run_setup = ":"
            run_invocation = (
                "bun src/main.ts run /tmp/task.md "
                f"{shlex.quote(NO_TRACE_OUT_DIR)} harness.json none"
            )
            report_invocation = ":"

        prepare_command = f"""
set -euo pipefail
printf '%s' {shlex.quote(encoded_instruction)} | base64 -d > /tmp/task.md
{WORKDIR_BOOTSTRAP}
agent_workdir="$(choose_agent_workdir /tmp/task.md)"
normalize_empty_app_alias "$agent_workdir"
printf '%s\n' "$agent_workdir" > {shlex.quote(WORKDIR_FILE)}
{trace_prepare}
"""
        await self.exec_as_root(
            environment,
            command=prepare_command,
            env=self._env,
            timeout_sec=60,
        )

        command = f"""
set -euo pipefail
{trace_run_setup}
cd {shlex.quote(APP_DIR)}
export AGENT_WORKDIR="$(cat {shlex.quote(WORKDIR_FILE)} 2>/dev/null || printf '%s' /app)"
run_status=0
agent_timeout="${{AGENT_RUN_TIMEOUT_SEC:-{DEFAULT_OUTER_AGENT_TIMEOUT_SEC}}}"
if command -v timeout >/dev/null 2>&1; then
  timeout "$agent_timeout" {run_invocation} || run_status=$?
else
  {run_invocation} || run_status=$?
fi
{report_invocation}
exit 0
"""
        await self.exec_as_agent(
            environment,
            command=command,
            env=self._env,
            timeout_sec=self._agent_run_timeout_sec() + 60,
        )
        context.metadata = {
            "agent": self.name(),
            "trace_mode": self._trace_mode,
            "harness_file": self._harness_file or "harness-runs/baseline/harness.json",
        }
        if self._trace_mode == "full":
            context.metadata["logs_dir"] = AGENT_LOG_DIR
