import path from "node:path";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { ChatOpenRouter } from "@langchain/openrouter";
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { createGeneratePptxTool } from "./generate-pptx-tool";
import { SYSTEM_PROMPT } from "./system-prompt";

export const AGENT_MODEL = "deepseek/deepseek-v4-flash";

export async function createSlideAgent(workspaceDir: string) {
  const model = new ChatOpenRouter({
    model: AGENT_MODEL,
  });

  const backend = await LocalShellBackend.create({
    rootDir: path.resolve(workspaceDir),
    virtualMode: true,
    inheritEnv: true,
  });

  const agent = createDeepAgent({
    model,
    systemPrompt: SYSTEM_PROMPT,
    tools: [createGeneratePptxTool(backend)],
    skills: ["./.agent/skills/"],
    memory: ["./AGENTS.md"],
    backend,
    checkpointer: new MemorySaver(),
  });

  return { agent, backend };
}
