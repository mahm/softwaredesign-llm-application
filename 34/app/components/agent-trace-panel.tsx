"use client";

import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  CircleDashed,
  TerminalSquare,
} from "lucide-react";
import { type AcpRunTrace, type AgentName, useTrace } from "./trace-context";

type AgentTracePanelProps = {
  agent: AgentName;
  emptyText: string;
};

function agentLabel(agent: string): string {
  if (agent === "claude") {
    return "Claude Code";
  }
  if (agent === "codex") {
    return "Codex";
  }
  return agent;
}

function turnLabel(turn?: number): string {
  return turn === undefined ? "単独呼び出し" : `ターン ${turn}`;
}

function turnOrder(turn?: number): number {
  return turn ?? 9999;
}

function statusIcon(status: AcpRunTrace["status"]) {
  if (status === "complete") {
    return (
      <CheckCircle2 className="size-4 text-emerald-600" aria-hidden="true" />
    );
  }
  if (status === "error") {
    return <AlertTriangle className="size-4 text-red-600" aria-hidden="true" />;
  }
  return <CircleDashed className="size-4 text-blue-600" aria-hidden="true" />;
}

function EventRows({ run }: { run: AcpRunTrace }) {
  if (run.events.length === 0) {
    return (
      <p className="mt-3 rounded-lg bg-slate-100 px-3 py-2 text-xs text-slate-500">
        まだACPイベントは届いていません。
      </p>
    );
  }

  return (
    <div className="mt-3 space-y-2">
      {run.events.map((event) => (
        <div
          key={
            event.id ??
            `${run.key}-${event.type}-${event.title}-${event.status ?? ""}-${event.text ?? ""}`
          }
          className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="min-w-0 truncate font-mono text-[11px] text-slate-700">
              {event.type}:{event.title}
            </span>
            {event.status ? (
              <span className="shrink-0 rounded bg-white px-1.5 py-0.5 text-[10px] text-slate-500">
                {event.status}
              </span>
            ) : null}
          </div>
          {event.text ? (
            <details className="mt-2">
              <summary className="cursor-pointer text-[11px] font-semibold text-slate-600">
                イベント本文を表示
              </summary>
              <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-white p-2 text-[11px] leading-relaxed text-slate-600">
                {event.text}
              </pre>
            </details>
          ) : null}
        </div>
      ))}
    </div>
  );
}

export function AgentTracePanel({ agent, emptyText }: AgentTracePanelProps) {
  const { runs } = useTrace();
  const agentRuns = runs
    .filter((run) => run.agent === agent)
    .slice()
    .sort(
      (left, right) =>
        turnOrder(left.turn) - turnOrder(right.turn) ||
        left.updatedAt - right.updatedAt,
    );

  return (
    <section className="min-h-full">
      {agentRuns.length > 0 ? (
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-medium text-slate-600">ACPイベント</h2>
          <TerminalSquare
            className="size-5 text-slate-400"
            aria-hidden="true"
          />
        </div>
      ) : null}

      {agentRuns.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <p className="text-center text-sm text-slate-400">{emptyText}</p>
        </div>
      ) : null}

      <div className="space-y-4">
        {agentRuns.map((run) => (
          <article
            key={run.key}
            className="rounded-lg border border-slate-200 bg-white p-3"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <Bot className="size-4 shrink-0 text-slate-500" />
                  <h3 className="truncate text-sm font-semibold text-slate-900">
                    {agentLabel(run.agent)}
                  </h3>
                  <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                    {turnLabel(run.turn)}
                  </span>
                  {statusIcon(run.status)}
                </div>
                <p className="mt-1 truncate font-mono text-[11px] text-slate-500">
                  {run.session}
                </p>
              </div>
              <div className="shrink-0 text-right text-[11px] text-slate-500">
                <div>{run.model ?? "model未取得"}</div>
                {run.reasoningEffort ? (
                  <div>effort={run.reasoningEffort}</div>
                ) : null}
                <div>read-only</div>
              </div>
            </div>
            {run.prompt ? (
              <details className="mt-3" open>
                <summary className="cursor-pointer text-xs font-semibold text-slate-600">
                  投入プロンプト全文
                </summary>
                <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-100 px-3 py-2 text-xs leading-relaxed text-slate-700">
                  {run.prompt}
                </pre>
              </details>
            ) : null}
            <EventRows run={run} />
            {run.finalText ? (
              <details className="mt-3" open>
                <summary className="cursor-pointer text-xs font-semibold text-slate-600">
                  このエージェントの応答
                </summary>
                <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-900 p-3 text-xs leading-relaxed text-slate-100">
                  {run.finalText}
                </pre>
              </details>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}
