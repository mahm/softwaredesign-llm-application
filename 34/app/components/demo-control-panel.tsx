"use client";

import { useCopilotChat } from "@copilotkit/react-core";
import { MessageRole, TextMessage } from "@copilotkit/runtime-client-gql";
import { Play, RotateCcw } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useTrace } from "./trace-context";

const CODEBASE_ANALYSIS_OBJECTIVE =
  "対象リポジトリのコードベースを分析し、概要、主要構成、処理の流れ、注意点を整理してください。";

function createClientId(): string {
  const webCrypto = globalThis.crypto;
  if (typeof webCrypto?.randomUUID === "function") {
    return webCrypto.randomUUID();
  }

  if (typeof webCrypto?.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    webCrypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
      "",
    );
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function buildPrompt(repoPath: string): {
  content: string;
  sessions: { claude: string; codex: string };
} {
  const runId = createClientId()
    .replace(/[^a-z0-9]/gi, "")
    .slice(0, 8);
  const sessions = {
    claude: `claude-${runId}`,
    codex: `codex-${runId}`,
  };

  return {
    sessions,
    content: `対象リポジトリ: ${repoPath}
調査目的: ${CODEBASE_ANALYSIS_OBJECTIVE}

ACP連携調査を開始してください。

- Claude Codeのセッション名は ${sessions.claude}
- Codexのセッション名は ${sessions.codex}
- まずask_acp_agentsをturn=1で1回呼び出し、Claude CodeとCodexを同時に起動してください
- 外部エージェントの応答に、対象リポジトリの調査に関係する確認質問が含まれている場合だけask_humanで利用者に確認してください
- ユーザーの回答後は、同じセッション名でturnを1増やしてask_acp_agentsを呼び出し、humanAnswerにはユーザー回答だけを入れてください
- 追加の確認質問が返った場合は、同じ手順でturnを増やして続けてください
- 外部エージェントの応答が十分に揃ったら、Claude CodeとCodexの見立てを比較し、統合レポートを簡潔に出してください`,
  };
}

export function DemoControlPanel() {
  const [repoPath, setRepoPath] = useState("");
  const { appendMessage, reset, isLoading } = useCopilotChat();
  const { resetTrace, updateSupervisor, upsertRun, watchAcpSessions } =
    useTrace();

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const { content, sessions } = buildPrompt(repoPath.trim());
    reset();
    resetTrace();
    watchAcpSessions([
      { agent: "claude", session: sessions.claude },
      { agent: "codex", session: sessions.codex },
    ]);
    upsertRun({
      key: `acpx:claude:${sessions.claude}:turn-1`,
      agent: "claude",
      session: sessions.claude,
      turn: 1,
      status: "pending",
      events: [],
    });
    upsertRun({
      key: `acpx:codex:${sessions.codex}:turn-1`,
      agent: "codex",
      session: sessions.codex,
      turn: 1,
      status: "pending",
      events: [],
    });
    updateSupervisor({
      prompt: content,
      status: "running",
      finalReport: undefined,
      errorSummary: undefined,
    });
    try {
      await appendMessage(
        new TextMessage({
          id: createClientId(),
          role: MessageRole.User,
          content,
        }),
      );
    } catch (error) {
      updateSupervisor({
        status: "error",
        errorSummary:
          error instanceof Error
            ? error.message
            : "Supervisorの呼び出しに失敗しました。",
      });
    }
  };

  const resetAll = () => {
    reset();
    resetTrace();
  };

  return (
    <section className="border-b border-slate-200 bg-slate-50 px-5 py-3">
      <form onSubmit={submit} className="grid gap-3">
        <div className="min-w-0">
          <label
            htmlFor="repoPath"
            className="mb-1 block text-xs font-medium text-slate-600"
          >
            対象リポジトリ
          </label>
          <input
            id="repoPath"
            value={repoPath}
            onChange={(event) => setRepoPath(event.target.value)}
            placeholder="/path/to/your/repository"
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-sm text-slate-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
          />
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <button
            type="submit"
            disabled={isLoading || !repoPath.trim()}
            className="inline-flex min-h-10 cursor-pointer items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition enabled:hover:-translate-y-px enabled:hover:bg-indigo-700 enabled:hover:shadow-md enabled:active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
          >
            <Play className="size-4" aria-hidden="true" />
            調査開始
          </button>
          <div className="group relative">
            <button
              type="button"
              onClick={resetAll}
              aria-label="会話とイベント表示をリセット"
              className="inline-flex min-h-10 cursor-pointer items-center justify-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-700 transition enabled:hover:-translate-y-px enabled:hover:border-slate-400 enabled:hover:bg-slate-100 enabled:hover:text-slate-900 enabled:hover:shadow-sm enabled:active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300"
            >
              <RotateCcw className="size-4" aria-hidden="true" />
            </button>
            <span className="pointer-events-none absolute right-0 top-full z-10 mt-2 w-max max-w-56 rounded-md bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-white opacity-0 shadow-lg transition group-hover:opacity-100 group-focus-within:opacity-100">
              会話とイベント表示をリセット
            </span>
          </div>
        </div>
      </form>
    </section>
  );
}
