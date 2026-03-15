"use client";

import { useCopilotChat } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { SlideProvider } from "./components/slide-context";
import { SlidePreview } from "./components/slide-preview";
import { ToolCallRenderer } from "./components/tool-call-renderer";

export default function Home() {
  const { isLoading } = useCopilotChat();

  return (
    <SlideProvider>
      <main className="flex h-screen bg-slate-50">
        {/* Left: Slide Preview */}
        <div className="flex w-1/2 flex-col border-r border-slate-200">
          <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
            <h1 className="text-base font-semibold tracking-tight text-slate-800">
              arXiv Slide Generator
            </h1>
          </header>
          <div className="flex-1 overflow-auto p-4">
            <SlidePreview />
          </div>
        </div>

        {/* Right: Chat */}
        <div className="flex w-1/2 flex-col">
          <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-5 py-3">
            <span className="text-sm font-medium text-slate-600">Chat</span>
            {isLoading && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700">
                <span className="inline-block size-1.5 animate-pulse rounded-full bg-indigo-500" />
                Processing
              </span>
            )}
          </header>
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <ToolCallRenderer />
            <CopilotChat
              labels={{
                title: "Slide Agent",
                initial:
                  "arXiv論文のURLを貼り付けてください。論文を分析してスライドを作成します。",
              }}
            />
          </div>
        </div>
      </main>
    </SlideProvider>
  );
}
