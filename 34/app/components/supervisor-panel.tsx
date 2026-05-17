"use client";

import { useCopilotChatInternal } from "@copilotkit/react-core";
import { AlertTriangle, CheckCircle2, CircleDashed } from "lucide-react";
import { useEffect, useMemo } from "react";
import { useTrace } from "./trace-context";

type MessageRecord = Record<string, unknown>;

type SupervisorEvent = {
  key: string;
  label: string;
  title: string;
  summary?: string;
  detail?: string;
  status?: string;
};

function asRecord(value: unknown): MessageRecord | undefined {
  if (value && typeof value === "object") {
    return value as MessageRecord;
  }
  return undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function normalizeValue(value: unknown): unknown {
  if (typeof value === "string") {
    return parseJson(value);
  }
  return value;
}

function textContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map(textContent).join("");
  }
  if (content && typeof content === "object" && "text" in content) {
    return textContent((content as { text?: unknown }).text);
  }
  return "";
}

function detailText(value: unknown): string {
  const normalized = normalizeValue(value);
  if (typeof normalized === "string") {
    return normalized;
  }
  try {
    return JSON.stringify(normalized, null, 2);
  } catch {
    return String(value);
  }
}

function shortText(text: string, max = 120): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= max) {
    return normalized;
  }
  return `${normalized.slice(0, max)}...`;
}

function statusText(record: MessageRecord): string | undefined {
  const status = asRecord(record.status);
  return asString(status?.code);
}

function toolName(toolCall: MessageRecord): string | undefined {
  const fn = asRecord(toolCall.function);
  return asString(fn?.name) ?? asString(toolCall.name);
}

function toolArguments(toolCall: MessageRecord): unknown {
  const fn = asRecord(toolCall.function);
  return fn?.arguments ?? toolCall.arguments;
}

function latestAssistantText(messages: unknown[]): string | undefined {
  return [...messages]
    .reverse()
    .map(
      (message) =>
        message as { role?: string; content?: unknown; toolCalls?: unknown[] },
    )
    .filter(
      (message) =>
        message.role === "assistant" &&
        !Array.isArray(message.toolCalls) &&
        textContent(message.content).trim().length > 0,
    )
    .map((message) => textContent(message.content).trim())
    .find(Boolean);
}

function toolCallSummary(name: string, args: unknown): string | undefined {
  const parsed = normalizeValue(args);
  const record = asRecord(parsed);
  if (!record) {
    return undefined;
  }

  if (name === "ask_acp_agents") {
    const sessions = asRecord(record.sessions);
    const parts = [
      `turn=${String(record.turn ?? "?")}`,
      sessions?.claude ? `Claude=${String(sessions.claude)}` : undefined,
      sessions?.codex ? `Codex=${String(sessions.codex)}` : undefined,
      record.humanAnswer ? "HITL回答を渡して再開" : undefined,
    ].filter(Boolean);
    return parts.join(" / ");
  }

  if (name === "ask_human") {
    return shortText(textContent(record.question));
  }

  return undefined;
}

function toolResultSummary(name: string | undefined, result: unknown): string {
  const parsed = normalizeValue(result);

  if (name === "ask_acp_agents") {
    const record = asRecord(parsed);
    const results = Array.isArray(record?.results) ? record.results : [];
    return `turn=${String(record?.turn ?? "?")} / ${results.length}件の外部エージェント結果`;
  }

  if (name === "ask_human") {
    if (textContent(parsed).trim() === "Awaiting approval") {
      return "利用者の回答待ち";
    }
    return `回答: ${shortText(textContent(parsed))}`;
  }

  return shortText(detailText(parsed));
}

function toolResultLabel(name: string | undefined, result: unknown): string {
  if (name !== "ask_human") {
    return "ツール結果";
  }
  return textContent(normalizeValue(result)).trim() === "Awaiting approval"
    ? "確認待ち"
    : "HITL回答";
}

function buildSupervisorEvents(messages: unknown[]): SupervisorEvent[] {
  const events: SupervisorEvent[] = [];
  const toolNames = new Map<string, string>();

  messages.forEach((message, messageIndex) => {
    const record = asRecord(message);
    if (!record) {
      return;
    }

    const rawId = asString(record.id);
    const keyBase = rawId
      ? `${rawId}:${messageIndex}`
      : `message-${messageIndex}`;
    const role = asString(record.role);
    const type = asString(record.type);
    const content = textContent(record.content).trim();

    if (role === "user" && content) {
      events.push({
        key: `${keyBase}:user`,
        label: "入力",
        title: "Supervisorへの入力",
        summary: shortText(content),
        detail: content,
        status: statusText(record),
      });
    }

    if (role === "assistant" && content && !Array.isArray(record.toolCalls)) {
      events.push({
        key: `${keyBase}:assistant`,
        label: "応答",
        title: "Supervisorの応答",
        summary: shortText(content),
        detail: content,
        status: statusText(record),
      });
    }

    const toolCalls = Array.isArray(record.toolCalls) ? record.toolCalls : [];
    toolCalls.forEach((toolCall, toolIndex) => {
      const tool = asRecord(toolCall);
      if (!tool) {
        return;
      }
      const id = asString(tool.id) ?? `${keyBase}:tool-${toolIndex}`;
      const name = toolName(tool) ?? "tool";
      const args = toolArguments(tool);
      toolNames.set(id, name);
      events.push({
        key: `${id}:call:${messageIndex}:${toolIndex}`,
        label: "ツール呼び出し",
        title: name,
        summary: toolCallSummary(name, args),
        detail: detailText(args),
        status: statusText(record),
      });
    });

    if (type === "ActionExecutionMessage") {
      const name = asString(record.name) ?? "tool";
      const args = record.arguments;
      toolNames.set(keyBase, name);
      events.push({
        key: `${keyBase}:call`,
        label: "ツール呼び出し",
        title: name,
        summary: toolCallSummary(name, args),
        detail: detailText(args),
        status: statusText(record),
      });
    }

    if (role === "tool" || type === "ResultMessage") {
      const toolCallId =
        asString(record.toolCallId) ?? asString(record.actionExecutionId);
      const name =
        asString(record.toolName) ??
        asString(record.actionName) ??
        (toolCallId ? toolNames.get(toolCallId) : undefined);
      const result = record.result ?? record.content;
      events.push({
        key: `${keyBase}:result`,
        label: toolResultLabel(name, result),
        title: name ?? "tool result",
        summary: toolResultSummary(name, result),
        detail: detailText(result),
        status: statusText(record),
      });
    }
  });

  return events;
}

function StatusPill({
  status,
  isLoading,
}: {
  status: string;
  isLoading: boolean;
}) {
  if (status === "complete") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
        <CheckCircle2 className="size-3.5" aria-hidden="true" />
        Complete
      </span>
    );
  }

  if (status === "error") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-red-200 bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700">
        <AlertTriangle className="size-3.5" aria-hidden="true" />
        Error
      </span>
    );
  }

  if (isLoading || status === "running") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700">
        <span className="inline-block size-1.5 animate-pulse rounded-full bg-indigo-500" />
        Processing
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs font-medium text-slate-500">
      <CircleDashed className="size-3.5" aria-hidden="true" />
      Idle
    </span>
  );
}

function SupervisorEvents({ events }: { events: SupervisorEvent[] }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-slate-900">
        Supervisorイベント
      </h2>
      {events.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-400">
          Supervisorの入力、ツール呼び出し、結果がここに表示されます。
        </p>
      ) : (
        <div className="space-y-2">
          {events.map((event) => (
            <article
              key={event.key}
              className="rounded-lg border border-slate-200 bg-slate-50 p-3"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded bg-white px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
                  {event.label}
                </span>
                <h3 className="min-w-0 flex-1 truncate font-mono text-[11px] font-semibold text-slate-800">
                  {event.title}
                </h3>
                {event.status ? (
                  <span className="rounded bg-white px-1.5 py-0.5 text-[10px] text-slate-500">
                    {event.status}
                  </span>
                ) : null}
              </div>
              {event.summary ? (
                <p className="mt-2 text-xs leading-relaxed text-slate-600">
                  {event.summary}
                </p>
              ) : null}
              {event.detail ? (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs font-semibold text-slate-600">
                    詳細
                  </summary>
                  <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded-md bg-white p-3 text-[11px] leading-relaxed text-slate-700">
                    {event.detail}
                  </pre>
                </details>
              ) : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function PromptCard({ prompt }: { prompt?: string }) {
  if (!prompt) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-center text-sm text-slate-400">
        対象リポジトリを入力すると、Supervisorへの投入プロンプトがここに表示されます。
      </div>
    );
  }

  return (
    <details
      className="rounded-lg border border-slate-200 bg-slate-50 p-3"
      open
    >
      <summary className="cursor-pointer text-xs font-semibold text-slate-600">
        投入プロンプト全文
      </summary>
      <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded-md bg-white p-3 text-[11px] leading-relaxed text-slate-700">
        {prompt}
      </pre>
    </details>
  );
}

function ErrorCard({ summary }: { summary?: string }) {
  if (!summary) {
    return null;
  }

  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
      <div className="mb-1 flex items-center gap-2 font-semibold">
        <AlertTriangle className="size-4" aria-hidden="true" />
        失敗の要約
      </div>
      <p className="leading-relaxed">{summary}</p>
    </div>
  );
}

function FinalReport({ report }: { report?: string }) {
  if (!report) {
    return null;
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-slate-900">
        Supervisorの最終レポート
      </h2>
      <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
        {report}
      </div>
    </section>
  );
}

export function SupervisorPanel() {
  const { interrupt, isLoading, messages } = useCopilotChatInternal();
  const { supervisor, updateSupervisor } = useTrace();
  const supervisorEvents = useMemo(
    () => buildSupervisorEvents(Array.isArray(messages) ? messages : []),
    [messages],
  );
  const assistantReport = latestAssistantText(
    Array.isArray(messages) ? messages : [],
  );
  const finalReport =
    supervisor.finalReport ?? (!isLoading ? assistantReport : undefined);
  const status = supervisor.errorSummary
    ? "error"
    : isLoading || supervisor.status === "running"
      ? "running"
      : finalReport
        ? "complete"
        : supervisor.status;

  useEffect(() => {
    if (isLoading || !assistantReport || supervisor.status === "complete") {
      return;
    }
    updateSupervisor({ finalReport: assistantReport, status: "complete" });
  }, [assistantReport, isLoading, supervisor.status, updateSupervisor]);

  return (
    <section className="grid gap-4 px-5 py-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-700">
          Supervisor実行状況
        </h2>
        <StatusPill status={status} isLoading={isLoading} />
      </div>
      <PromptCard prompt={supervisor.prompt} />
      <SupervisorEvents events={supervisorEvents} />
      <ErrorCard summary={supervisor.errorSummary} />
      {interrupt ? <div className="w-full">{interrupt}</div> : null}
      <FinalReport report={finalReport} />
    </section>
  );
}
