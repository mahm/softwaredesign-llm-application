import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { ChatOpenRouter } from "@langchain/openrouter";
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { config as loadDotenv } from "dotenv";
import { createMiddleware } from "langchain";

export const DEFAULT_MODEL = "deepseek/deepseek-v4-flash";
export const DEFAULT_HARNESS_FILE = "harness-runs/baseline/harness.json";
export const DEFAULT_WORKING_DIRECTORY = "/app";
export const DEFAULT_TEMPERATURE = 1;
export const DEFAULT_RECURSION_LIMIT = 150;
export const DEFAULT_COMMAND_TIMEOUT_SEC = 180;
export const DEFAULT_AGENT_RUN_TIMEOUT_SEC = 840;
export const DEFAULT_MAX_OUTPUT_BYTES = 160_000;

export type HarnessConfig = {
  systemPrompt: string | string[];
  model?: string;
  temperature?: number;
  recursionLimit?: number;
  commandTimeoutSec?: number;
  agentRunTimeoutSec?: number;
  maxOutputBytes?: number;
};

export type CreateAgentOptions = {
  harness?: HarnessConfig;
  harnessFile?: string;
  traceDirectory?: string;
  workingDirectory?: string;
};

type ResolvedAgentOptions = EffectiveHarnessConfig & {
  traceDirectory?: string;
};

export type EffectiveHarnessConfig = {
  systemPrompt: string | string[];
  model: string;
  temperature: number;
  recursionLimit: number;
  commandTimeoutSec: number;
  agentRunTimeoutSec: number;
  maxOutputBytes: number;
  workingDirectory: string;
};

export type TerminalAgentResult = {
  result: unknown;
  finalText: string;
  threadId: string;
  model: string;
  temperature: number;
  recursionLimit: number;
  commandTimeoutSec: number;
  agentRunTimeoutSec: number;
  maxOutputBytes: number;
  workingDirectory: string;
  startedAt: string;
  durationMs: number;
  harness: HarnessConfig;
  harnessFile?: string;
  effectiveHarness: EffectiveHarnessConfig;
};

export type TerminalAgentInstance = {
  agent: any;
  backend: Awaited<ReturnType<typeof LocalShellBackend.create>>;
  checkpointer: MemorySaver;
  threadId: string;
  model: string;
  temperature: number;
  recursionLimit: number;
  commandTimeoutSec: number;
  agentRunTimeoutSec: number;
  maxOutputBytes: number;
  workingDirectory: string;
  traceDirectory?: string;
  startedAt: string;
  startedAtMs: number;
  harness: HarnessConfig;
  harnessFile?: string;
  effectiveHarness: EffectiveHarnessConfig;
};

type CheckpointSnapshotSource = {
  checkpointer: MemorySaver;
  threadId: string;
};

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
}

function writeJson(file: string, value: unknown) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function appendJsonl(file: string, value: unknown) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, `${JSON.stringify(value)}\n`, "utf-8");
}

function toJsonValue(value: unknown, depth = 0): unknown {
  if (depth > 10) {
    return "[MaxDepth]";
  }

  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === "string") {
    return value.length > 10_000 ? `${value.slice(0, 10_000)}...[truncated]` : value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }

  if (typeof value === "bigint") {
    return value.toString();
  }

  if (value instanceof Uint8Array) {
    return {
      type: "Uint8Array",
      base64: Buffer.from(value).toString("base64"),
    };
  }

  if (Array.isArray(value)) {
    return value.map((item) => toJsonValue(item, depth + 1));
  }

  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        toJsonValue(item, depth + 1),
      ])
    );
  }

  return String(value);
}

async function writeCheckpointSnapshot(
  checkpointer: MemorySaver,
  threadId: string,
  outDir: string
) {
  const checkpoints = [];

  for await (const checkpoint of checkpointer.list({
    configurable: { thread_id: threadId },
  })) {
    checkpoints.push(toJsonValue(checkpoint));
  }

  writeJson(path.join(outDir, "checkpoints.json"), {
    threadId,
    checkpointCount: checkpoints.length,
    checkpoints,
  });
}

async function safeWriteCheckpointSnapshot(
  checkpointer: MemorySaver,
  threadId: string,
  outDir?: string
) {
  if (!outDir) {
    return;
  }

  try {
    await writeCheckpointSnapshot(checkpointer, threadId, outDir);
  } catch (error: any) {
    writeJson(path.join(outDir, "checkpoints.error.json"), {
      error: String(error?.message ?? error),
    });
  }
}

function createTraceMiddleware(
  outDir: string,
  snapshotSource: CheckpointSnapshotSource
) {
  let completedToolCalls = 0;
  let lastCheckpointFlushMs = 0;

  async function flushCheckpointSnapshot(force = false) {
    const now = Date.now();
    if (!force && completedToolCalls % 5 !== 0 && now - lastCheckpointFlushMs < 30_000) {
      return;
    }

    lastCheckpointFlushMs = now;
    await safeWriteCheckpointSnapshot(
      snapshotSource.checkpointer,
      snapshotSource.threadId,
      outDir
    );
  }

  return createMiddleware({
    name: "TraceMiddleware",
    wrapToolCall: async (request: any, handler: any) => {
      const startedAt = Date.now();
      const toolName = request.toolCall?.name ?? "unknown";
      const args = request.toolCall?.args ?? {};

      try {
        const result = await handler(request);
        appendJsonl(path.join(outDir, "tool_events.jsonl"), {
          ts: new Date().toISOString(),
          toolName,
          args,
          ok: true,
          durationMs: Date.now() - startedAt,
          contentLength:
            typeof result?.content === "string" ? result.content.length : undefined,
          status: result?.status,
        });
        completedToolCalls += 1;
        await flushCheckpointSnapshot();
        return result;
      } catch (error: any) {
        appendJsonl(path.join(outDir, "tool_events.jsonl"), {
          ts: new Date().toISOString(),
          toolName,
          args,
          ok: false,
          durationMs: Date.now() - startedAt,
          error: String(error?.message ?? error),
        });
        completedToolCalls += 1;
        await flushCheckpointSnapshot(true);
        throw error;
      }
    },
  });
}

export function extractMessageText(message: any): string {
  const content = message?.content;

  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (typeof part?.text === "string") {
          return part.text;
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }

  return "";
}

export function resolveTerminalAgentWorkingDirectory(
  workingDirectory = process.env.AGENT_WORKDIR ?? DEFAULT_WORKING_DIRECTORY
): string {
  return path.resolve(workingDirectory);
}

function loadHarness(file = DEFAULT_HARNESS_FILE): HarnessConfig {
  return readJson<HarnessConfig>(file);
}

function resolveCreateAgentOptions(
  options: CreateAgentOptions = {}
): {
  agentOptions: ResolvedAgentOptions;
  harness: HarnessConfig;
  harnessFile?: string;
  effectiveHarness: EffectiveHarnessConfig;
} {
  if (options.harness && options.harnessFile) {
    throw new Error("Pass either harness or harnessFile, not both.");
  }

  const harnessFile = options.harness ? undefined : options.harnessFile ?? DEFAULT_HARNESS_FILE;
  const harness = options.harness ?? loadHarness(harnessFile);
  const effectiveHarness: EffectiveHarnessConfig = {
    systemPrompt: harness.systemPrompt,
    model: process.env.BASE_MODEL ?? harness.model ?? DEFAULT_MODEL,
    temperature: harness.temperature ?? DEFAULT_TEMPERATURE,
    recursionLimit: harness.recursionLimit ?? DEFAULT_RECURSION_LIMIT,
    commandTimeoutSec: harness.commandTimeoutSec ?? DEFAULT_COMMAND_TIMEOUT_SEC,
    agentRunTimeoutSec:
      harness.agentRunTimeoutSec ?? DEFAULT_AGENT_RUN_TIMEOUT_SEC,
    maxOutputBytes: harness.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES,
    workingDirectory: resolveTerminalAgentWorkingDirectory(options.workingDirectory),
  };

  return {
    agentOptions: {
      ...effectiveHarness,
      traceDirectory: options.traceDirectory,
    },
    harness,
    harnessFile,
    effectiveHarness,
  };
}

export async function createAgent(
  options: CreateAgentOptions = {}
): Promise<TerminalAgentInstance> {
  const { agentOptions, harness, harnessFile, effectiveHarness } =
    resolveCreateAgentOptions(options);

  if (!process.env.OPENROUTER_API_KEY) {
    throw new Error("OPENROUTER_API_KEY is required. Put it in 35/.env or export it.");
  }

  const startedAtMs = Date.now();
  const startedAt = new Date(startedAtMs).toISOString();
  const threadId = crypto.randomUUID();
  const systemPrompt = Array.isArray(agentOptions.systemPrompt)
    ? agentOptions.systemPrompt.join("\n")
    : agentOptions.systemPrompt;

  fs.mkdirSync(agentOptions.workingDirectory, { recursive: true });
  if (agentOptions.traceDirectory) {
    fs.mkdirSync(agentOptions.traceDirectory, { recursive: true });
    fs.writeFileSync(path.join(agentOptions.traceDirectory, "thread_id.txt"), threadId, "utf-8");
  }

  const model = new ChatOpenRouter({
    model: agentOptions.model,
    temperature: agentOptions.temperature,
  });
  const checkpointer = new MemorySaver();
  const backend = await LocalShellBackend.create({
    rootDir: agentOptions.workingDirectory,
    timeout: agentOptions.commandTimeoutSec,
    maxOutputBytes: agentOptions.maxOutputBytes,
    inheritEnv: true,
  });
  const middleware = agentOptions.traceDirectory
    ? [createTraceMiddleware(agentOptions.traceDirectory, { checkpointer, threadId })]
    : [];

  const agent = createDeepAgent({
    model,
    backend,
    systemPrompt,
    checkpointer,
    middleware: middleware as any,
  });

  return {
    agent,
    backend,
    checkpointer,
    threadId,
    model: agentOptions.model,
    temperature: agentOptions.temperature,
    recursionLimit: agentOptions.recursionLimit,
    commandTimeoutSec: agentOptions.commandTimeoutSec,
    agentRunTimeoutSec: agentOptions.agentRunTimeoutSec,
    maxOutputBytes: agentOptions.maxOutputBytes,
    workingDirectory: agentOptions.workingDirectory,
    traceDirectory: agentOptions.traceDirectory,
    startedAt,
    startedAtMs,
    harness,
    harnessFile,
    effectiveHarness,
  };
}

function withAgentRunTimeout<T>(
  operation: Promise<T>,
  agentRunTimeoutSec?: number
): Promise<T> {
  if (!agentRunTimeoutSec || agentRunTimeoutSec <= 0) {
    return operation;
  }

  let timeout: NodeJS.Timeout | undefined;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => {
      reject(
        new Error(`Agent run timed out after ${agentRunTimeoutSec} seconds`)
      );
    }, agentRunTimeoutSec * 1000);
  });

  return Promise.race([operation, timeoutPromise]).finally(() => {
    if (timeout) {
      clearTimeout(timeout);
    }
  });
}

export async function runAgent(
  instance: TerminalAgentInstance,
  task: string
): Promise<TerminalAgentResult> {
  try {
    const result = await withAgentRunTimeout(
      instance.agent.invoke(
        {
          messages: [
            {
              role: "user",
              content: task,
            },
          ],
        },
        {
          recursionLimit: instance.recursionLimit,
          configurable: {
            thread_id: instance.threadId,
          },
        }
      ),
      instance.agentRunTimeoutSec
    );

    const messages = Array.isArray((result as any).messages)
      ? (result as any).messages
      : [];
    const finalMessage = messages[messages.length - 1];

    await safeWriteCheckpointSnapshot(
      instance.checkpointer,
      instance.threadId,
      instance.traceDirectory
    );

    return {
      result,
      finalText: extractMessageText(finalMessage),
      threadId: instance.threadId,
      model: instance.model,
      temperature: instance.temperature,
      recursionLimit: instance.recursionLimit,
      commandTimeoutSec: instance.commandTimeoutSec,
      agentRunTimeoutSec: instance.agentRunTimeoutSec,
      maxOutputBytes: instance.maxOutputBytes,
      workingDirectory: instance.workingDirectory,
      startedAt: instance.startedAt,
      durationMs: Date.now() - instance.startedAtMs,
      harness: instance.harness,
      harnessFile: instance.harnessFile,
      effectiveHarness: instance.effectiveHarness,
    };
  } catch (error) {
    await safeWriteCheckpointSnapshot(
      instance.checkpointer,
      instance.threadId,
      instance.traceDirectory
    );
    throw error;
  } finally {
    await instance.backend.close();
  }
}

function printStandaloneUsage() {
  console.error(
    [
      "Usage: bun src/agent.ts run [task-file|-] [harness-file] [trace-dir]",
      "",
      "Examples:",
      "  bun src/agent.ts run /tmp/task.md harness-runs/baseline/harness.json",
      "  printf 'Create /tmp/answer.txt' | bun src/agent.ts run - harness-runs/baseline/harness.json results/manual-agent",
    ].join("\n")
  );
}

async function runStandaloneCli() {
  loadDotenv({ path: new URL("../.env", import.meta.url).pathname, quiet: true });

  const command = process.argv[2];
  if (command !== "run") {
    printStandaloneUsage();
    process.exit(1);
  }

  const taskFile = process.argv[3] ?? "-";
  const harnessFile = process.argv[4] ?? DEFAULT_HARNESS_FILE;
  const traceDirectory = process.argv[5];
  const task =
    taskFile === "-" ? fs.readFileSync(0, "utf-8") : fs.readFileSync(taskFile, "utf-8");

  const instance = await createAgent({
    harnessFile,
    traceDirectory,
  });

  if (traceDirectory) {
    writeJson(path.join(traceDirectory, "harness.used.json"), instance.effectiveHarness);
  }

  const result = await runAgent(instance, task);

  if (traceDirectory) {
    writeJson(path.join(traceDirectory, "final.json"), result.result);
  }

  process.stdout.write(result.finalText);
  if (!result.finalText.endsWith("\n")) {
    process.stdout.write("\n");
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runStandaloneCli().catch((error) => {
    console.error(String(error?.message ?? error));
    process.exit(1);
  });
}
