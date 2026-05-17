import path from "node:path";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { askAcpAgents, askHuman } from "./acpx";
import { SYSTEM_PROMPT } from "./system-prompt";

const workspaceDir = path.resolve(process.cwd(), "workspace");

const globalForAcpSupervisor = globalThis as typeof globalThis & {
  __acpSupervisorCheckpointer?: MemorySaver;
};

function supervisorCheckpointer(): MemorySaver {
  globalForAcpSupervisor.__acpSupervisorCheckpointer ??= new MemorySaver();
  return globalForAcpSupervisor.__acpSupervisorCheckpointer;
}

export function createAcpSupervisorAgent() {
  const model = new ChatOpenAI({
    model: "gpt-5.5",
    reasoning: { effort: "low" },
    useResponsesApi: true,
  });

  const backend = new LocalShellBackend({
    rootDir: workspaceDir,
    virtualMode: true,
    inheritEnv: true,
  });

  return createDeepAgent({
    name: "acpx_supervisor",
    model,
    systemPrompt: SYSTEM_PROMPT,
    tools: [askAcpAgents, askHuman],
    skills: ["./.agent/skills/"],
    memory: ["./AGENTS.md"],
    backend,
    checkpointer: supervisorCheckpointer(),
  });
}
