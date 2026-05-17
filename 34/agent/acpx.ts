import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { access } from "node:fs/promises";
import path from "node:path";
import { interrupt } from "@langchain/langgraph";
import { tool } from "langchain";
import { z } from "zod";
import { writeTraceSnapshot } from "./trace-store";

const ACP_AGENTS = ["claude", "codex"] as const;

type AcpAgentName = (typeof ACP_AGENTS)[number];

type AcpxRunResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
};

type AcpxEvent = {
  id: string;
  type: "message" | "thought" | "tool" | "usage" | "system" | "error";
  title: string;
  text?: string;
  status?: string;
};

type AcpxToolResult = {
  agent: AcpAgentName;
  session: string;
  turn?: number;
  model: string;
  acpxModel: string;
  reasoningEffort?: string;
  permissionMode: string;
  commandOptions: string[];
  prompt: string;
  promptPreview: string;
  finalText: string;
  events: AcpxEvent[];
  stderr?: string;
};

const AGENT_CONFIG: Record<
  AcpAgentName,
  { model: string; acpxModel: string; reasoningEffort?: "low" }
> = {
  claude: { model: "sonnet", acpxModel: "sonnet" },
  codex: { model: "gpt-5.5", acpxModel: "gpt-5.5/low", reasoningEffort: "low" },
};

const READ_ONLY_ACPX_ARGS = [
  "--approve-reads",
  "--non-interactive-permissions",
  "deny",
  "--no-terminal",
] as const;

function acpxExecutable(): string {
  return process.platform === "win32" ? "acpx.cmd" : "acpx";
}

async function assertDirectory(repoPath: string): Promise<string> {
  const absolute = path.resolve(repoPath);
  await access(absolute);
  return absolute;
}

function runAcpx(
  args: string[],
  stdinText = "",
  killAfterMs = 20 * 60_000,
  onStdoutLine?: (line: string) => void,
): Promise<AcpxRunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(acpxExecutable(), args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });

    let stdout = "";
    let stderr = "";
    let pendingStdout = "";
    let killedByTimer = false;

    const timer = setTimeout(() => {
      killedByTimer = true;
      child.kill("SIGTERM");
    }, killAfterMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      if (onStdoutLine) {
        pendingStdout += chunk;
        const lines = pendingStdout.split(/\r?\n/);
        pendingStdout = lines.pop() ?? "";
        for (const line of lines) {
          onStdoutLine(line);
        }
      }
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });

    child.on("close", (exitCode, signal) => {
      clearTimeout(timer);
      if (onStdoutLine && pendingStdout.trim()) {
        onStdoutLine(pendingStdout);
      }
      if (killedByTimer) {
        reject(
          new Error(
            `acpx timed out after ${Math.round(killAfterMs / 1000)}s: acpx ${args.join(" ")}`,
          ),
        );
        return;
      }
      resolve({ stdout, stderr, exitCode, signal });
    });

    child.stdin.end(stdinText);
  });
}

function failIfAcpxFailed(command: string[], result: AcpxRunResult): string {
  if (result.exitCode === 0) {
    return result.stdout.trim();
  }

  const detail = [result.stderr.trim(), result.stdout.trim()]
    .filter(Boolean)
    .join("\n");
  throw new Error(
    [
      `acpx command failed: acpx ${command.join(" ")}`,
      `exitCode=${result.exitCode} signal=${result.signal ?? "none"}`,
      detail,
    ]
      .filter(Boolean)
      .join("\n"),
  );
}

function commonArgs(cwd: string, timeoutSec: number): string[] {
  return ["--cwd", cwd, "--timeout", String(timeoutSec)];
}

async function runAcpxQuiet(params: {
  agent: AcpAgentName;
  cwd: string;
  args: string[];
  timeoutSec: number;
}): Promise<string> {
  const args = [
    ...commonArgs(params.cwd, params.timeoutSec),
    "--format",
    "quiet",
    params.agent,
    ...params.args,
  ];
  const result = await runAcpx(args, "", (params.timeoutSec + 30) * 1000);
  return failIfAcpxFailed(args, result);
}

async function configureSession(params: {
  agent: AcpAgentName;
  cwd: string;
  session: string;
  timeoutSec: number;
}): Promise<void> {
  const config = AGENT_CONFIG[params.agent];

  await runAcpxQuiet({
    ...params,
    args: ["sessions", "ensure", "--name", params.session],
  });
  await runAcpxQuiet({
    ...params,
    args: ["set", "model", config.acpxModel, "-s", params.session],
  });

  if (config.reasoningEffort && config.acpxModel === config.model) {
    await runAcpxQuiet({
      ...params,
      args: [
        "set",
        "reasoning_effort",
        config.reasoningEffort,
        "-s",
        params.session,
      ],
    });
  }
}

function truncate(text: string, max = 1200): string {
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}...`;
}

function contentToText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map(contentToText).filter(Boolean).join("\n");
  }
  if (content && typeof content === "object") {
    const record = content as Record<string, unknown>;
    if (typeof record.text === "string") {
      return record.text;
    }
    if ("content" in record) {
      return contentToText(record.content);
    }
  }
  return "";
}

class AcpxEventAccumulator {
  private readonly events: AcpxEvent[] = [];
  private readonly messageChunks: string[] = [];
  private nextEventId = 0;

  appendLine(line: string): boolean {
    if (!line.trim()) {
      return false;
    }

    let message: any;
    try {
      message = JSON.parse(line);
    } catch {
      this.pushEvent({
        type: "system",
        title: "non-json output",
        text: line,
      });
      return true;
    }

    if (message.error) {
      this.pushEvent({
        type: "error",
        title: "ACP error",
        text: message.error.message ?? JSON.stringify(message.error, null, 2),
      });
      return true;
    }

    if (message.method === "session/request_permission") {
      const toolCall = message.params?.toolCall;
      this.pushEvent({
        type: "system",
        title: "ACP HITL: permission request",
        text: JSON.stringify(
          {
            title: toolCall?.title,
            kind: toolCall?.kind,
            options: message.params?.options,
          },
          null,
          2,
        ),
      });
      return true;
    }

    const permissionEscalation =
      message.result?._meta?.acpx?.permissionEscalation;
    if (permissionEscalation) {
      this.pushEvent({
        type: "system",
        title: "ACP HITL: permission escalation",
        text: JSON.stringify(permissionEscalation, null, 2),
      });
      return true;
    }

    if (message.method !== "session/update") {
      if (message.method) {
        this.pushEvent({
          type: "system",
          title: String(message.method),
          text: JSON.stringify(message.params ?? message.result ?? {}, null, 2),
        });
      } else if (message.result?.stopReason) {
        this.pushEvent({
          type: "system",
          title: "turn finished",
          status: String(message.result.stopReason),
        });
      }
      return true;
    }

    const update = message.params?.update;
    const kind = update?.sessionUpdate;

    if (kind === "agent_message_chunk") {
      const text = contentToText(update.content);
      if (text) {
        this.messageChunks.push(text);
        this.appendTextEvent("message", "agent message", text);
      }
      return true;
    }

    if (kind === "agent_thought_chunk") {
      const text = contentToText(update.content);
      if (text) {
        this.appendTextEvent("thought", "reasoning", text);
      }
      return true;
    }

    if (kind === "tool_call" || kind === "tool_call_update") {
      const title = update.title ?? update.kind ?? update.toolCallId ?? "tool";
      const output = update.rawOutput ?? update.content ?? update.rawInput;
      const outputText = output
        ? contentToText(output) || JSON.stringify(output, null, 2)
        : undefined;
      this.pushEvent({
        type: "tool",
        title: String(title),
        status: update.status ? String(update.status) : undefined,
        text: outputText,
      });
      return true;
    }

    if (kind === "usage_update") {
      this.pushEvent({
        type: "usage",
        title: "usage",
        text: JSON.stringify(update, null, 2),
      });
      return true;
    }

    this.pushEvent({
      type: "system",
      title: String(kind ?? "session/update"),
      text: JSON.stringify(update ?? {}, null, 2),
    });
    return true;
  }

  snapshot(): { finalText: string; events: AcpxEvent[] } {
    return {
      finalText: this.messageChunks.join("").trim(),
      events: this.events.slice(-80),
    };
  }

  private appendTextEvent(
    type: "message" | "thought",
    title: string,
    text: string,
  ) {
    const last = this.events.at(-1);
    if (last?.type === type && last.title === title) {
      last.text = `${last.text ?? ""}${text}`;
      return;
    }
    this.pushEvent({
      type,
      title,
      text,
    });
  }

  private pushEvent(event: Omit<AcpxEvent, "id">) {
    this.nextEventId += 1;
    this.events.push({
      id: `event-${this.nextEventId}`,
      ...event,
    });
  }
}

function parseAcpxJson(stdout: string): {
  finalText: string;
  events: AcpxEvent[];
} {
  const accumulator = new AcpxEventAccumulator();

  for (const line of stdout.split(/\r?\n/)) {
    accumulator.appendLine(line);
  }

  return accumulator.snapshot();
}

function buildResearchPrompt(params: {
  agent: AcpAgentName;
  cwd: string;
  objective: string;
  turn: number;
  humanAnswer?: string;
}): string {
  if (params.turn > 1) {
    const answer = params.humanAnswer?.trim();
    if (!answer) {
      throw new Error("2ターン目以降はhumanAnswerが必要です。");
    }
    return answer;
  }

  return `あなたは対象リポジトリを調査するコーディングエージェントです。
対象リポジトリ: ${params.cwd}
調査目的: ${params.objective}

制約:
- 必ず読み取り専用で調査してください。
- ファイル編集、生成、削除、フォーマット、依存関係変更など、リポジトリに変更を加える操作は禁止です。
- サブエージェント、内部Task、長時間のバックグラウンド調査は使わず、現在のセッション内で読み取り調査を完結してください。
- この調査を起動している画面や呼び出し側の都合は調査対象ではありません。
- 確認質問は、対象リポジトリのコード、使い方、調査範囲に直接関係する内容だけにしてください。
- 後続ターンで利用者からの回答だけが届いた場合は、同じ調査の続きとして扱ってください。

まず対象リポジトリを読み、最終分析に入る前に不明点や出力粒度の確認事項を1つ以上質問してください。
質問には、読んだものと現時点の仮説を短く添えてください。
利用者から回答が届いたら、その回答を前提に追加調査を行い、必要なら追加質問、十分なら最終分析を返してください。

最終分析の出力形式:
- 調査目的に対する結論
- 根拠となるファイルや実装
- 注意点
- 次に読むとよい場所`;
}

async function promptAcpAgent(params: {
  agent: AcpAgentName;
  cwd: string;
  session: string;
  turn?: number;
  prompt: string;
  timeoutSec: number;
}): Promise<AcpxToolResult> {
  const config = AGENT_CONFIG[params.agent];
  await configureSession(params);
  const promptPreview = truncate(params.prompt, 260);
  const accumulator = new AcpxEventAccumulator();
  let writeQueue = Promise.resolve();
  let lastRunningTraceAt = 0;

  const enqueueTrace = (
    status: "running" | "complete" | "error",
    extra?: { finalText?: string; stderr?: string },
  ) => {
    const parsed = accumulator.snapshot();
    // ACPXのJSONLは少しずつ届くため、イベントをまとめながらtraceへ書き出す。
    writeQueue = writeQueue
      .then(() =>
        writeTraceSnapshot({
          agent: params.agent,
          session: params.session,
          turn: params.turn,
          model: config.model,
          reasoningEffort: config.reasoningEffort,
          prompt: params.prompt,
          promptPreview,
          status,
          finalText: extra?.finalText ?? parsed.finalText,
          events:
            status === "error" && extra?.stderr
              ? [
                  ...parsed.events,
                  {
                    type: "error",
                    title: "ACPX error",
                    text: extra.stderr,
                  },
                ]
              : parsed.events,
        }),
      )
      .catch(() => {});
  };
  const enqueueRunningTrace = () => {
    const now = Date.now();
    if (now - lastRunningTraceAt < 300) {
      return;
    }
    lastRunningTraceAt = now;
    enqueueTrace("running");
  };

  const args = [
    ...commonArgs(params.cwd, params.timeoutSec),
    "--format",
    "json",
    "--json-strict",
    "--model",
    config.acpxModel,
    ...READ_ONLY_ACPX_ARGS,
    params.agent,
    "-s",
    params.session,
    "--file",
    "-",
  ];
  enqueueTrace("running");
  const result = await runAcpx(
    args,
    params.prompt,
    (params.timeoutSec + 60) * 1000,
    (line) => {
      if (accumulator.appendLine(line)) {
        enqueueRunningTrace();
      }
    },
  );
  let stdout = "";
  try {
    stdout = failIfAcpxFailed(args, result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    enqueueTrace("error", { stderr: message });
    await writeQueue;
    throw error;
  }
  const parsed = accumulator.snapshot();
  const finalText =
    parsed.finalText || "(ACPエージェントから最終メッセージが返りませんでした)";
  if (!parsed.finalText && stdout) {
    const fallbackParsed = parseAcpxJson(stdout);
    parsed.events = fallbackParsed.events;
  }
  enqueueTrace("complete", { finalText });
  await writeQueue;

  return {
    agent: params.agent,
    session: params.session,
    turn: params.turn,
    model: config.model,
    acpxModel: config.acpxModel,
    reasoningEffort: config.reasoningEffort,
    permissionMode: "read-only",
    commandOptions: [...READ_ONLY_ACPX_ARGS],
    prompt: params.prompt,
    promptPreview,
    finalText,
    events: parsed.events,
    stderr: result.stderr.trim() ? result.stderr.trim() : undefined,
  };
}

export const askAcpAgents = tool(
  async ({
    repoPath,
    objective,
    turn,
    sessions,
    humanAnswer,
    timeoutSec = 900,
  }) => {
    const cwd = await assertDirectory(repoPath);
    const sessionMap = sessions as Record<AcpAgentName, string>;
    const requests = ACP_AGENTS.map((agent) => ({
      agent,
      session: sessionMap[agent],
      prompt: buildResearchPrompt({
        agent,
        cwd,
        objective,
        turn,
        humanAnswer,
      }),
    }));

    const results = await Promise.all(
      requests.map((request) =>
        promptAcpAgent({
          agent: request.agent,
          cwd,
          session: request.session,
          turn,
          prompt: request.prompt,
          timeoutSec,
        }),
      ),
    );

    return JSON.stringify({
      turn,
      humanAnswer,
      permissionMode: "read-only",
      commandOptions: [...READ_ONLY_ACPX_ARGS],
      results,
    });
  },
  {
    name: "ask_acp_agents",
    description:
      "ACPXを使い、Claude CodeとCodexへ同じ調査を並列に依頼する。同じセッション名とturn番号で会話を継続できる。",
    schema: z.object({
      repoPath: z.string().min(1).describe("調査対象リポジトリのパス"),
      objective: z.string().min(1).describe("調査目的"),
      turn: z
        .number()
        .int()
        .positive()
        .describe("同じセッションでの会話ターン番号。初回は1"),
      sessions: z
        .object({
          claude: z
            .string()
            .min(1)
            .regex(/^[a-zA-Z0-9_.:-]+$/),
          codex: z
            .string()
            .min(1)
            .regex(/^[a-zA-Z0-9_.:-]+$/),
        })
        .describe("Claude CodeとCodexのACPX名前付きセッション"),
      humanAnswer: z
        .string()
        .optional()
        .describe(
          "2ターン目以降で外部エージェントへ渡す利用者回答。この文字列だけを送る。",
        ),
      timeoutSec: z.number().int().positive().max(3600).optional().default(900),
    }),
  },
);

export const askAcpAgent = tool(
  async ({ agent, session, repoPath, prompt, timeoutSec = 900 }) => {
    const cwd = await assertDirectory(repoPath);
    const result = await promptAcpAgent({
      agent,
      cwd,
      session,
      turn: undefined,
      prompt,
      timeoutSec,
    });
    return JSON.stringify(result);
  },
  {
    name: "ask_acp_agent",
    description:
      "ACPXを使い、Claude CodeまたはCodexにリポジトリ調査用のプロンプトを送る。セッション名を同じにすると会話が続く。",
    schema: z.object({
      agent: z.enum(ACP_AGENTS).describe("呼び出すACP対応エージェント"),
      session: z
        .string()
        .min(1)
        .regex(/^[a-zA-Z0-9_.:-]+$/)
        .describe("ACPXの名前付きセッション"),
      repoPath: z.string().min(1).describe("調査対象リポジトリのパス"),
      prompt: z.string().min(1).describe("外部エージェントへ送るプロンプト"),
      timeoutSec: z.number().int().positive().max(3600).optional().default(900),
    }),
  },
);

export const askHuman = tool(
  async ({ question, context }) => {
    return interrupt<
      { kind: "human_question"; question: string; context?: string },
      string
    >({
      kind: "human_question",
      question,
      context,
    });
  },
  {
    name: "ask_human",
    description:
      "Claude CodeやCodexから出た確認事項を、人間の利用者にまとめて質問する。",
    schema: z.object({
      question: z.string().min(1).describe("利用者に確認したい質問"),
      context: z.string().optional().describe("この回答が必要な理由"),
    }),
  },
);

export function createThreadId(): string {
  return `acpx-demo-${randomUUID().slice(0, 8)}`;
}
