import path from "node:path";
import { ChatAnthropic } from "@langchain/anthropic";
import { MemorySaver } from "@langchain/langgraph";
import { LocalShellBackend, createDeepAgent } from "deepagents";
import { createGeneratePptxTool } from "./generate-pptx-tool";
import { SYSTEM_PROMPT } from "./system-prompt";

const workspaceDir = path.resolve(process.cwd(), "workspace");

export function createSlideAgent() {
  const model = new ChatAnthropic({
    model: "claude-sonnet-4-6",
  });

  const backend = new LocalShellBackend({
    rootDir: workspaceDir,
    virtualMode: true,
    inheritEnv: true,
  });

  return createDeepAgent({
    model,
    systemPrompt: SYSTEM_PROMPT,
    tools: [createGeneratePptxTool(backend)],
    skills: ["./.agent/skills/"],
    memory: ["./AGENTS.md"],
    backend,
    checkpointer: new MemorySaver(),
  });
}
