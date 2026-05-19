"use client";

import { useCopilotChat } from "@copilotkit/react-core";
import { AgentTracePanel } from "./components/agent-trace-panel";
import { DemoControlPanel } from "./components/demo-control-panel";
import { HumanQuestionInterrupt } from "./components/human-question-interrupt";
import { SupervisorPanel } from "./components/supervisor-panel";
import { TraceProvider } from "./components/trace-context";

export default function Home() {
  const { isLoading } = useCopilotChat();

  return (
    <TraceProvider>
      <main className="grid h-screen w-full max-w-full grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)] overflow-hidden bg-slate-50">
        <div className="flex min-h-0 min-w-0 flex-col border-r border-slate-200">
          <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
            <div className="min-w-0">
              <h1 className="text-base font-semibold tracking-tight text-slate-800">
                Claude Code
              </h1>
              <p className="truncate text-xs text-slate-500">
                sonnet / read-only
              </p>
            </div>
          </header>
          <div className="min-h-0 flex-1 overflow-auto p-4">
            <AgentTracePanel
              agent="claude"
              emptyText="調査を開始すると、ここにClaude CodeのACPイベントが表示されます"
            />
          </div>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col border-r border-slate-200">
          <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
            <div className="min-w-0">
              <h1 className="text-base font-semibold tracking-tight text-slate-800">
                Codex
              </h1>
              <p className="truncate text-xs text-slate-500">
                gpt-5.5 / low / read-only
              </p>
            </div>
          </header>
          <div className="min-h-0 flex-1 overflow-auto p-4">
            <AgentTracePanel
              agent="codex"
              emptyText="調査を開始すると、ここにCodexのACPイベントが表示されます"
            />
          </div>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col">
          <header className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-5 py-3">
            <span className="min-w-0 text-sm font-medium text-slate-600">
              Supervisor
            </span>
            {isLoading ? (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700">
                <span className="inline-block size-1.5 animate-pulse rounded-full bg-indigo-500" />
                Processing
              </span>
            ) : null}
          </header>
          <div className="min-h-0 flex-1 overflow-auto bg-white">
            <DemoControlPanel />
            <HumanQuestionInterrupt />
            <SupervisorPanel />
          </div>
        </div>
      </main>
    </TraceProvider>
  );
}
