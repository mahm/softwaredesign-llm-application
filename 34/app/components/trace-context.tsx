"use client";

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export type AgentName = "claude" | "codex" | string;

export type AcpxEvent = {
  id?: string;
  type: "message" | "thought" | "tool" | "usage" | "system" | "error";
  title: string;
  text?: string;
  status?: string;
};

export type AcpRunTrace = {
  key: string;
  agent: AgentName;
  session: string;
  turn?: number;
  model?: string;
  reasoningEffort?: string;
  prompt?: string;
  promptPreview?: string;
  status: "pending" | "running" | "complete" | "error";
  finalText?: string;
  events: AcpxEvent[];
  updatedAt: number;
};

export type HumanQuestionTrace = {
  key: string;
  question: string;
  context?: string;
  answer?: string;
  status: "pending" | "answered";
  updatedAt: number;
};

export type SupervisorTrace = {
  prompt?: string;
  finalReport?: string;
  errorSummary?: string;
  status: "idle" | "running" | "complete" | "error";
  updatedAt: number;
};

export type WatchedAcpSession = {
  agent: "claude" | "codex";
  session: string;
};

type TraceContextValue = {
  runs: AcpRunTrace[];
  humanQuestions: HumanQuestionTrace[];
  supervisor: SupervisorTrace;
  upsertRun: (run: Omit<AcpRunTrace, "updatedAt">) => void;
  upsertHumanQuestion: (
    question: Omit<HumanQuestionTrace, "updatedAt">,
  ) => void;
  updateSupervisor: (
    supervisor: Partial<Omit<SupervisorTrace, "updatedAt">>,
  ) => void;
  watchAcpSessions: (sessions: WatchedAcpSession[]) => void;
  resetTrace: () => void;
};

const TraceContext = createContext<TraceContextValue | null>(null);

function equalData(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function runData(
  run: Omit<AcpRunTrace, "updatedAt">,
): Omit<AcpRunTrace, "updatedAt"> {
  return {
    key: run.key,
    agent: run.agent,
    session: run.session,
    turn: run.turn,
    model: run.model,
    reasoningEffort: run.reasoningEffort,
    prompt: run.prompt,
    promptPreview: run.promptPreview,
    status: run.status,
    finalText: run.finalText,
    events: run.events,
  };
}

function humanQuestionData(
  question: Omit<HumanQuestionTrace, "updatedAt">,
): Omit<HumanQuestionTrace, "updatedAt"> {
  return {
    key: question.key,
    question: question.question,
    context: question.context,
    answer: question.answer,
    status: question.status,
  };
}

export function TraceProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<AcpRunTrace[]>([]);
  const [humanQuestions, setHumanQuestions] = useState<HumanQuestionTrace[]>(
    [],
  );
  const [watchedSessions, setWatchedSessions] = useState<WatchedAcpSession[]>(
    [],
  );
  const [supervisor, setSupervisor] = useState<SupervisorTrace>({
    status: "idle",
    updatedAt: Date.now(),
  });

  const upsertRun = useCallback((run: Omit<AcpRunTrace, "updatedAt">) => {
    setRuns((current) => {
      const nextData = runData(run);
      const index = current.findIndex((item) => item.key === run.key);
      if (index === -1) {
        return [...current, { ...nextData, updatedAt: Date.now() }];
      }
      if (equalData(runData(current[index]), nextData)) {
        return current;
      }
      return current.map((item, itemIndex) =>
        itemIndex === index
          ? { ...item, ...nextData, updatedAt: Date.now() }
          : item,
      );
    });
  }, []);

  const upsertHumanQuestion = useCallback(
    (question: Omit<HumanQuestionTrace, "updatedAt">) => {
      setHumanQuestions((current) => {
        const nextData = humanQuestionData(question);
        const index = current.findIndex((item) => item.key === question.key);
        if (index === -1) {
          return [...current, { ...nextData, updatedAt: Date.now() }];
        }
        if (equalData(humanQuestionData(current[index]), nextData)) {
          return current;
        }
        return current.map((item, itemIndex) =>
          itemIndex === index
            ? { ...item, ...nextData, updatedAt: Date.now() }
            : item,
        );
      });
    },
    [],
  );

  const resetTrace = useCallback(() => {
    setRuns([]);
    setHumanQuestions([]);
    setWatchedSessions([]);
    setSupervisor({
      status: "idle",
      updatedAt: Date.now(),
    });
  }, []);

  const updateSupervisor = useCallback(
    (nextSupervisor: Partial<Omit<SupervisorTrace, "updatedAt">>) => {
      setSupervisor((current) => {
        const next = { ...current, ...nextSupervisor };
        const currentData = { ...current, updatedAt: undefined };
        const nextData = { ...next, updatedAt: undefined };
        if (equalData(currentData, nextData)) {
          return current;
        }
        return {
          ...next,
          updatedAt: Date.now(),
        };
      });
    },
    [],
  );

  const watchAcpSessions = useCallback((sessions: WatchedAcpSession[]) => {
    setWatchedSessions((current) => {
      const next = [...current];
      for (const session of sessions) {
        if (
          !next.some(
            (item) =>
              item.agent === session.agent && item.session === session.session,
          )
        ) {
          next.push(session);
        }
      }
      return next.length === current.length ? current : next;
    });
  }, []);

  useEffect(() => {
    if (
      watchedSessions.length === 0 ||
      supervisor.status === "complete" ||
      supervisor.status === "error"
    ) {
      return;
    }

    let cancelled = false;
    const poll = async () => {
      for (const session of watchedSessions) {
        try {
          const searchParams = new URLSearchParams({
            agent: session.agent,
            session: session.session,
          });

          const response = await fetch(
            `/api/acpx-traces?${searchParams.toString()}`,
            { cache: "no-store" },
          );
          if (!response.ok) {
            continue;
          }
          const body = (await response.json()) as {
            found?: boolean;
            snapshot?: Omit<AcpRunTrace, "key">;
            snapshots?: Omit<AcpRunTrace, "key">[];
          };
          if (cancelled || !body.found) {
            continue;
          }
          const snapshots =
            body.snapshots ?? (body.snapshot ? [body.snapshot] : []);
          for (const snapshot of snapshots) {
            // ACPXは別プロセスで動くため、画面側ではtraceファイルを読み直して進行中の状態を拾う。
            upsertRun({
              key: `acpx:${snapshot.agent}:${snapshot.session}:turn-${snapshot.turn ?? "single"}`,
              agent: snapshot.agent,
              session: snapshot.session,
              model: snapshot.model,
              reasoningEffort: snapshot.reasoningEffort,
              turn: snapshot.turn,
              prompt: snapshot.prompt,
              promptPreview: snapshot.promptPreview,
              status: snapshot.status,
              finalText: snapshot.finalText,
              events: snapshot.events,
            });
          }
        } catch {
          // trace取得に失敗しても、最終状態はツール結果から反映できるためここでは止めない。
        }
      }
    };

    poll();
    const timer = window.setInterval(poll, 500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [supervisor.status, upsertRun, watchedSessions]);

  const value = useMemo(
    () => ({
      runs,
      humanQuestions,
      supervisor,
      upsertRun,
      upsertHumanQuestion,
      updateSupervisor,
      watchAcpSessions,
      resetTrace,
    }),
    [
      runs,
      humanQuestions,
      supervisor,
      upsertRun,
      upsertHumanQuestion,
      updateSupervisor,
      watchAcpSessions,
      resetTrace,
    ],
  );

  return (
    <TraceContext.Provider value={value}>{children}</TraceContext.Provider>
  );
}

export function useTrace() {
  const value = useContext(TraceContext);
  if (!value) {
    throw new Error("TraceProvider is missing");
  }
  return value;
}
