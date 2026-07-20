import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadDotenv } from "dotenv";
import { AGENT_MODEL, createSlideAgent } from "../agent/agent";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
loadDotenv({ path: path.join(ROOT, ".env"), quiet: true });

const [arxivId, variant = "baseline"] = process.argv.slice(2);
if (!arxivId) {
  console.error(
    "Usage: bun agent-run/run.ts <arxivId> [baseline|improvement-1|improvement-2]",
  );
  process.exit(1);
}

const workspaceDir = path.join(ROOT, "workspaces", variant);
const { agent, backend } = await createSlideAgent(workspaceDir);

const config = {
  recursionLimit: 150,
  configurable: { thread_id: crypto.randomUUID() },
};

// サブエージェント内部を含む全ツール呼び出しを評価用に記録する
const toolCalls: { name: string; args: unknown }[] = [];

function extractMessageText(message: any): string {
  const content = message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : (part?.text ?? "")))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

async function runTurn(content: string): Promise<string> {
  console.log(`\n[user] ${content}`);
  const events = agent.streamEvents(
    { messages: [{ role: "user", content }] },
    { ...config, version: "v2" },
  );
  for await (const event of events) {
    if (event.event === "on_tool_start") {
      toolCalls.push({ name: event.name, args: event.data?.input ?? {} });
      console.log(`[tool] ${event.name}`);
    }
  }
  const state: any = await agent.getState(config);
  const text = extractMessageText(state.values.messages.at(-1));
  console.log(text);
  return text;
}

const startedAtMs = Date.now();

// ターン1: 論文URLを渡し、アウトライン提案まで進める
const request = `https://arxiv.org/abs/${arxivId} この論文からスライドを作成してください。`;
await runTurn(request);

// ターン2: 人間の確認を固定の承認メッセージで置き換え、generate_pptxまで進める
const approval = "OKです。この構成でスライドを生成してください。";
const finalText = await runTurn(approval);

await backend.close();

const slides = JSON.parse(
  fs.readFileSync(path.join(workspaceDir, "slides", `${arxivId}.json`), "utf-8"),
);

const resultFile = path.join(ROOT, "results", variant, `${arxivId}.json`);
fs.mkdirSync(path.dirname(resultFile), { recursive: true });
fs.writeFileSync(
  resultFile,
  `${JSON.stringify(
    {
      arxivId,
      variant,
      model: AGENT_MODEL,
      messages: [request, approval],
      toolCalls,
      slides,
      finalText,
      durationMs: Date.now() - startedAtMs,
      generatedAt: new Date().toISOString(),
    },
    null,
    2,
  )}\n`,
);
console.log(`\n[saved] results/${variant}/${arxivId}.json`);
