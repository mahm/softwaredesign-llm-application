import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

export type StoredAcpxEvent = {
  id?: string;
  type: "message" | "thought" | "tool" | "usage" | "system" | "error";
  title: string;
  text?: string;
  status?: string;
};

export type StoredTraceSnapshot = {
  agent: string;
  session: string;
  turn?: number;
  model?: string;
  reasoningEffort?: string;
  prompt?: string;
  promptPreview?: string;
  status: "pending" | "running" | "complete" | "error";
  finalText?: string;
  events: StoredAcpxEvent[];
  updatedAt: number;
};

const traceDir = path.join(
  tmpdir(),
  "softwaredesign-llm-application-34",
  "acpx-traces",
);

function safeName(value: string): string {
  return value.replace(/[^a-zA-Z0-9_.:-]/g, "_");
}

function tracePath(agent: string, session: string, turn?: number): string {
  return path.join(
    traceDir,
    `${safeName(agent)}__${safeName(session)}__${safeName(turn === undefined ? "single" : `turn-${turn}`)}.json`,
  );
}

export async function writeTraceSnapshot(
  snapshot: Omit<StoredTraceSnapshot, "updatedAt">,
): Promise<void> {
  await mkdir(traceDir, { recursive: true });
  await writeFile(
    tracePath(snapshot.agent, snapshot.session, snapshot.turn),
    `${JSON.stringify({ ...snapshot, updatedAt: Date.now() })}\n`,
    "utf8",
  );
}

export async function readTraceSnapshot(
  agent: string,
  session: string,
  turn?: number,
): Promise<StoredTraceSnapshot | null> {
  try {
    const content = await readFile(tracePath(agent, session, turn), "utf8");
    return JSON.parse(content) as StoredTraceSnapshot;
  } catch {
    return null;
  }
}

export async function readTraceSnapshots(
  agent: string,
  session: string,
): Promise<StoredTraceSnapshot[]> {
  try {
    const prefix = `${safeName(agent)}__${safeName(session)}__`;
    const filenames = await readdir(traceDir);
    const snapshots = await Promise.all(
      filenames
        .filter((filename) => filename.startsWith(prefix))
        .map(async (filename) => {
          const content = await readFile(path.join(traceDir, filename), "utf8");
          return JSON.parse(content) as StoredTraceSnapshot;
        }),
    );

    return snapshots.sort(
      (left, right) =>
        (left.turn ?? 9999) - (right.turn ?? 9999) ||
        left.updatedAt - right.updatedAt,
    );
  } catch {
    return [];
  }
}
