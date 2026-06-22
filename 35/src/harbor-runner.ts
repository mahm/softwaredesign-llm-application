import fs from "node:fs";
import path from "node:path";

import {
  DEFAULT_HARNESS_FILE,
  createAgent,
  extractMessageText,
  runAgent,
} from "./agent.js";

export { DEFAULT_HARNESS_FILE } from "./agent.js";

export type TraceMode = "full" | "none";

function resolveTraceMode(value = "full"): TraceMode {
  if (value === "full" || value === "none") {
    return value;
  }

  throw new Error(`trace mode must be "full" or "none": ${value}`);
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
}

function writeJson(file: string, value: unknown) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function readJsonl(file: string): any[] {
  if (!fs.existsSync(file)) {
    return [];
  }

  return fs
    .readFileSync(file, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export async function runHarborTask(
  taskFile = "/tmp/task.md",
  outDir = "results/manual",
  harnessFile = DEFAULT_HARNESS_FILE,
  traceModeValue = "full"
) {
  const traceMode = resolveTraceMode(traceModeValue);
  const shouldTrace = traceMode === "full";
  const startedAtMs = Date.now();
  const startedAt = new Date(startedAtMs).toISOString();
  let agent: Awaited<ReturnType<typeof createAgent>> | undefined;

  if (shouldTrace) {
    fs.mkdirSync(outDir, { recursive: true });
    writeJson(path.join(outDir, "run.json"), {
      harnessFile,
      startedAt,
      status: "started",
      traceMode,
    });
  }

  try {
    const task = fs.readFileSync(taskFile, "utf-8");
    agent = await createAgent({
      harnessFile,
      traceDirectory: shouldTrace ? outDir : undefined,
    });
    if (shouldTrace) {
      writeJson(path.join(outDir, "harness.used.json"), agent.effectiveHarness);
      writeJson(path.join(outDir, "run.json"), {
        harnessFile,
        traceMode,
        model: agent.model,
        temperature: agent.temperature,
        recursionLimit: agent.recursionLimit,
        commandTimeoutSec: agent.commandTimeoutSec,
        agentRunTimeoutSec: agent.agentRunTimeoutSec,
        maxOutputBytes: agent.maxOutputBytes,
        workingDirectory: agent.workingDirectory,
        startedAt,
        status: "started",
      });
    }

    const result = await runAgent(agent, task);

    if (shouldTrace) {
      writeJson(path.join(outDir, "final.json"), result.result);
      writeJson(path.join(outDir, "run.json"), {
        harnessFile,
        traceMode,
        model: result.model,
        temperature: result.temperature,
        recursionLimit: result.recursionLimit,
        commandTimeoutSec: result.commandTimeoutSec,
        agentRunTimeoutSec: result.agentRunTimeoutSec,
        maxOutputBytes: result.maxOutputBytes,
        workingDirectory: result.workingDirectory,
        startedAt,
        status: "completed",
        durationMs: Date.now() - startedAtMs,
        finalText: result.finalText,
      });
    }
  } catch (error: any) {
    if (shouldTrace) {
      writeJson(path.join(outDir, "run.json"), {
        harnessFile,
        traceMode,
        model: agent?.model,
        temperature: agent?.temperature,
        recursionLimit: agent?.recursionLimit,
        commandTimeoutSec: agent?.commandTimeoutSec,
        agentRunTimeoutSec: agent?.agentRunTimeoutSec,
        maxOutputBytes: agent?.maxOutputBytes,
        workingDirectory: agent?.workingDirectory,
        startedAt,
        status: "failed",
        durationMs: Date.now() - startedAtMs,
        error: String(error?.message ?? error),
      });
    }
    throw error;
  }
}

export function reportRun(outDir = "results/manual") {
  const runFile = path.join(outDir, "run.json");
  const toolEvents = readJsonl(path.join(outDir, "tool_events.jsonl"));
  const final = fs.existsSync(path.join(outDir, "final.json"))
    ? readJson<any>(path.join(outDir, "final.json"))
    : null;
  let runInfo = fs.existsSync(runFile)
    ? readJson<any>(runFile)
    : null;
  const reward = fs.existsSync(path.join(outDir, "reward.json"))
    ? readJson<any>(path.join(outDir, "reward.json"))
    : null;

  if (runInfo?.status === "started" && !final) {
    const startedAt =
      typeof runInfo.startedAt === "string" ? Date.parse(runInfo.startedAt) : NaN;
    runInfo = {
      ...runInfo,
      status: "interrupted",
      durationMs: Number.isFinite(startedAt) ? Date.now() - startedAt : undefined,
    };
    writeJson(runFile, runInfo);
  }

  const executeEvents = toolEvents.filter((e) => e.toolName === "execute");
  const editEvents = toolEvents.filter((e) => /edit|write/i.test(e.toolName));
  const failedEvents = toolEvents.filter((e) => !e.ok);

  const commands = executeEvents.map((e) => e.args?.command ?? e.args);
  const finalMessages = Array.isArray(final?.messages) ? final.messages : [];
  const finalText =
    extractMessageText(finalMessages[finalMessages.length - 1]) ||
    String(runInfo?.finalText ?? "");

  const markdown = [
    "# Analysis input",
    "",
    "## Reward",
    "```json",
    JSON.stringify(reward, null, 2),
    "```",
    "",
    "## Tool event summary",
    `- tool calls: ${toolEvents.length}`,
    `- execute calls: ${executeEvents.length}`,
    `- edit/write calls: ${editEvents.length}`,
    `- failed tool calls: ${failedEvents.length}`,
    "",
    "## Commands observed",
    "```json",
    JSON.stringify(commands.slice(0, 50), null, 2),
    "```",
    "",
    "## Failed tool events",
    "```json",
    JSON.stringify(failedEvents.slice(0, 20), null, 2),
    "```",
    "",
    "## Final answer",
    "```text",
    finalText.slice(0, 12000),
    "```",
  ].join("\n");

  fs.writeFileSync(path.join(outDir, "analysis_input.md"), markdown, "utf-8");
  writeJson(path.join(outDir, "summary.json"), {
    toolCalls: toolEvents.length,
    executeCalls: executeEvents.length,
    editCalls: editEvents.length,
    failedToolCalls: failedEvents.length,
  });
}
