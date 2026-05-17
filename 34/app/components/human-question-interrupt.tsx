"use client";

import { useLangGraphInterrupt } from "@copilotkit/react-core";
import { Send } from "lucide-react";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import { useTrace } from "./trace-context";

type HumanQuestionPayload = {
  kind: "human_question";
  question: string;
  context?: string;
};

function tinyHash(text: string): string {
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = Math.imul(31, hash) + text.charCodeAt(index);
  }
  return Math.abs(hash).toString(36);
}

function HumanQuestionForm({
  payload,
  resolve,
}: {
  payload: HumanQuestionPayload;
  resolve: (answer: string) => void;
}) {
  const [answer, setAnswer] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const { upsertHumanQuestion } = useTrace();
  const key = useMemo(
    () => `human:${tinyHash(payload.question + (payload.context ?? ""))}`,
    [payload.context, payload.question],
  );

  useEffect(() => {
    upsertHumanQuestion({
      key,
      question: payload.question,
      context: payload.context,
      status: "pending",
    });
  }, [key, payload.context, payload.question, upsertHumanQuestion]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = answer.trim();
    if (!trimmed || submitted) {
      return;
    }
    setSubmitted(true);
    upsertHumanQuestion({
      key,
      question: payload.question,
      context: payload.context,
      answer: trimmed,
      status: "answered",
    });
    resolve(trimmed);
  };

  return (
    <form
      onSubmit={submit}
      className="w-full rounded-lg border border-amber-200 bg-amber-50 p-4"
    >
      <div className="mb-3">
        <p className="text-xs font-medium text-amber-700">ユーザー確認</p>
        <h2 className="mt-1 text-base font-semibold text-slate-900">
          外部エージェントからの確認
        </h2>
      </div>
      {payload.context ? (
        <p className="mb-3 rounded border border-amber-200 bg-white px-3 py-2 text-sm text-slate-700">
          {payload.context}
        </p>
      ) : null}
      <pre className="mb-3 max-h-52 overflow-auto whitespace-pre-wrap rounded bg-white p-3 text-sm leading-relaxed text-slate-900">
        {payload.question}
      </pre>
      <textarea
        value={answer}
        onChange={(event) => setAnswer(event.target.value)}
        rows={4}
        className="w-full resize-none rounded border border-amber-300 bg-white px-3 py-2 text-sm leading-relaxed text-slate-900 outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-200"
        placeholder="ここに回答を入力します"
      />
      <div className="mt-3 flex justify-end">
        <button
          type="submit"
          disabled={!answer.trim() || submitted}
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition enabled:hover:-translate-y-px enabled:hover:bg-indigo-700 enabled:hover:shadow-md enabled:active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
        >
          <Send className="size-4" aria-hidden="true" />
          回答して再開
        </button>
      </div>
    </form>
  );
}

export function HumanQuestionInterrupt() {
  useLangGraphInterrupt<HumanQuestionPayload>({
    enabled: ({ eventValue }) => eventValue?.kind === "human_question",
    render: ({ event, resolve }) => (
      <HumanQuestionForm payload={event.value} resolve={resolve} />
    ),
  });

  return null;
}
