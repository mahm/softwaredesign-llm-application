import { tool } from "@langchain/core/tools";
import { ChatOpenRouter } from "@langchain/openrouter";
import { createDeepAgent, FilesystemBackend } from "deepagents";
import { cp, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

import { getOpenRouterConfig } from "./openrouter";

const root = fileURLToPath(new URL("..", import.meta.url));
const workspace = path.join(root, ".workspaces", "deepagents");

await rm(workspace, { recursive: true, force: true });
await mkdir(path.dirname(workspace), { recursive: true });
await cp(path.join(root, "workspace"), workspace, { recursive: true });

async function runTests() {
  const child = Bun.spawn(["bun", "test"], {
    cwd: workspace,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  return { exitCode, output: `${stdout}${stderr}`.trim() };
}

const runTestsTool = tool(
  async () => (await runTests()).output,
  {
    name: "run_tests",
    description: "現在のプロジェクトでbun testを実行します。",
    schema: z.object({}),
  },
);

const { apiKey, model, provider } = getOpenRouterConfig();
const chatModel = new ChatOpenRouter({
  apiKey,
  model,
  temperature: 0,
  provider,
});

const agent = createDeepAgent({
  model: chatModel,
  backend: new FilesystemBackend({ rootDir: workspace, virtualMode: true }),
  tools: [runTestsTool],
});

const result = await agent.invoke(
  {
    messages: [
      {
        role: "user",
        content: "requirements.mdを確認して開発してください。",
      },
    ],
  },
  { recursionLimit: 40 },
);

const tests = await runTests();
if (tests.exitCode !== 0) {
  throw new Error(`Tests still fail after the agent run:\n${tests.output}`);
}

console.log(result.messages.at(-1)?.content);
console.log(tests.output);
