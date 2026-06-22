WORKDIR_FILE = "/tmp/deepagents-agent-workdir"

WORKDIR_BOOTSTRAP = r"""
choose_agent_workdir() {
  task_file="${1:-/tmp/task.md}"

  if [ -n "${AGENT_WORKDIR:-}" ] && [ -d "$AGENT_WORKDIR" ]; then
    printf '%s\n' "$AGENT_WORKDIR"
    return 0
  fi

  if grep -Eq '(^|[^[:alnum:]_/-])/app($|[^[:alnum:]_/-])|/app/' "$task_file" 2>/dev/null && [ -d /app ]; then
    printf '%s\n' /app
    return 0
  fi

  if grep -Eq '(^|[^[:alnum:]_/-])/workspace($|[^[:alnum:]_/-])|/workspace/' "$task_file" 2>/dev/null && [ -d /workspace ]; then
    printf '%s\n' /workspace
    return 0
  fi

  for candidate in /workspace /app; do
    if [ -d "$candidate" ] && find "$candidate" -mindepth 1 -maxdepth 3 -print -quit 2>/dev/null | grep -q .; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  for candidate in /workspace /app; do
    if [ -d "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  printf '%s\n' /app
}

normalize_empty_app_alias() {
  chosen="$1"

  if [ "$chosen" != /workspace ] || [ ! -d /workspace ] || [ ! -d /app ] || [ -L /app ]; then
    return 0
  fi

  if find /app -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi

  rmdir /app 2>/dev/null && ln -s /workspace /app 2>/dev/null || true
}
"""
