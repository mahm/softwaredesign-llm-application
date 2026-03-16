"use client";

import { useDefaultTool } from "@copilotkit/react-core";
import type { CatchAllActionRenderProps } from "@copilotkit/react-core";
import PptxGenJS from "pptxgenjs";
import { useEffect, useRef } from "react";
import { type SlideData, useSlideData } from "./slide-context";

const colors = {
  primary: "1B2A4A",
  accent: "2E86AB",
  text: "333333",
  white: "FFFFFF",
};

async function generatePptxBase64(data: SlideData): Promise<string> {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.title = data.title;
  if (data.author) pptx.author = data.author;

  for (const s of data.slides) {
    const slide = pptx.addSlide();

    if (s.type === "title") {
      slide.background = { color: colors.primary };
      slide.addText(s.title, {
        x: 0.8,
        y: 1.5,
        w: "85%",
        h: 1.5,
        fontSize: 36,
        color: colors.white,
        bold: true,
        align: "left",
      });
      if (s.subtitle) {
        slide.addText(s.subtitle, {
          x: 0.8,
          y: 3.2,
          w: "85%",
          h: 0.8,
          fontSize: 18,
          color: "B0BEC5",
          align: "left",
        });
      }
    } else if (s.type === "section") {
      slide.background = { color: colors.accent };
      slide.addText(s.title, {
        x: 0.8,
        y: 2.0,
        w: "85%",
        h: 1.5,
        fontSize: 32,
        color: colors.white,
        bold: true,
        align: "left",
      });
    } else if (s.type === "content") {
      slide.addText(s.title, {
        x: 0.5,
        y: 0.3,
        w: "90%",
        h: 0.8,
        fontSize: 28,
        color: colors.primary,
        bold: true,
      });
      slide.addShape("rect", {
        x: 0.5,
        y: 1.0,
        w: "90%",
        h: 0.03,
        fill: { color: colors.accent },
      });
      if (s.bullets && s.bullets.length > 0) {
        const bulletText = s.bullets.map((b) => ({
          text: b,
          options: {
            fontSize: 16,
            color: colors.text,
            bullet: { code: "2022" },
            breakLine: true,
            paraSpaceAfter: 8,
          },
        }));
        slide.addText(bulletText, {
          x: 0.8,
          y: 1.3,
          w: "85%",
          h: 4.0,
          valign: "top",
        });
      }
    }
  }

  return (await pptx.write({ outputType: "base64" })) as string;
}

function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={`inline-block size-3 rounded-full border-2 border-current border-t-transparent animate-spin ${className ?? ""}`}
    />
  );
}

function DownloadCard() {
  const { pptxBase64 } = useSlideData();

  if (!pptxBase64) return null;

  const handleDownload = () => {
    const binary = atob(pptxBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], {
      type: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "presentation.pptx";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="my-2 rounded-xl border border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50 p-4">
      <div className="flex items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-lg bg-emerald-100">
          <svg
            className="size-5 text-emerald-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <title>Download</title>
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <div className="flex-1">
          <p className="text-sm font-semibold text-emerald-900">
            スライドの生成が完了しました
          </p>
          <p className="text-xs text-emerald-600">presentation.pptx</p>
        </div>
        <button
          type="button"
          onClick={handleDownload}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-emerald-700"
        >
          ダウンロード
        </button>
      </div>
    </div>
  );
}

function GeneratePptxHandler({
  result,
  status,
}: { result: unknown; status: string }) {
  const { setSlideData, setPptxBase64 } = useSlideData();
  const generatedRef = useRef(false);

  useEffect(() => {
    if (status !== "complete" || generatedRef.current) return;

    const parsed = typeof result === "string" ? JSON.parse(result) : result;
    if (!parsed?.success) return;

    generatedRef.current = true;

    const data: SlideData = {
      title: (parsed.title as string) ?? "",
      author: parsed.author as string | undefined,
      slides: (parsed.slides as SlideData["slides"]) ?? [],
    };

    setSlideData(data);
    generatePptxBase64(data).then(setPptxBase64);
  }, [status, result, setSlideData, setPptxBase64]);

  if (status === "complete") {
    return <DownloadCard />;
  }

  return (
    <div className="my-1 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 font-mono text-[13px]">
      <div className="flex items-center gap-1.5">
        <Spinner className="text-amber-500" />
        <strong className="text-amber-900">generate_pptx</strong>
        <span className="text-xs text-amber-600">generating</span>
      </div>
    </div>
  );
}

export function ToolCallRenderer() {
  useDefaultTool(
    {
      render: (props: CatchAllActionRenderProps) => {
        const { status, name, args } = props;

        if (name === "generate_pptx") {
          return <GeneratePptxHandler result={props.result} status={status} />;
        }

        if (status === "inProgress") {
          return (
            <div className="my-1 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 font-mono text-[13px]">
              <div className="flex items-center gap-1.5">
                <Spinner className="text-indigo-500" />
                <strong className="text-indigo-900">{name}</strong>
              </div>
              {Object.keys(args).length > 0 && (
                <pre className="mt-1 whitespace-pre-wrap break-all text-[11px] text-slate-500">
                  {JSON.stringify(args, null, 2)}
                </pre>
              )}
            </div>
          );
        }

        if (status === "executing") {
          return (
            <div className="my-1 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 font-mono text-[13px]">
              <div className="flex items-center gap-1.5">
                <Spinner className="text-amber-500" />
                <strong className="text-amber-900">{name}</strong>
                <span className="text-xs text-amber-600">executing</span>
              </div>
              <pre className="mt-1 whitespace-pre-wrap break-all text-[11px] text-slate-500">
                {JSON.stringify(args, null, 2)}
              </pre>
            </div>
          );
        }

        const hasResult =
          props.result !== undefined &&
          props.result !== null &&
          props.result !== "";
        const resultStr = hasResult
          ? JSON.stringify(props.result, null, 2)
          : null;
        const isLong = resultStr != null && resultStr.length > 200;

        return (
          <div className="my-1 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 font-mono text-[13px]">
            <div className="flex items-center gap-1.5">
              <svg
                className="size-3.5 text-emerald-600"
                viewBox="0 0 16 16"
                fill="currentColor"
              >
                <title>Complete</title>
                <path
                  fillRule="evenodd"
                  d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14Zm3.28-8.72a.75.75 0 0 0-1.06-1.06L7 8.44 5.78 7.22a.75.75 0 0 0-1.06 1.06l1.75 1.75a.75.75 0 0 0 1.06 0l3.75-3.75Z"
                  clipRule="evenodd"
                />
              </svg>
              <strong className="text-emerald-900">{name}</strong>
            </div>
            {isLong ? (
              <details className="mt-1">
                <summary className="cursor-pointer text-[11px] text-slate-500 hover:text-slate-700">
                  Show result
                </summary>
                <pre className="mt-1 whitespace-pre-wrap break-all text-[11px] text-slate-700">
                  {resultStr}
                </pre>
              </details>
            ) : resultStr ? (
              <pre className="mt-1 whitespace-pre-wrap break-all text-[11px] text-slate-700">
                {resultStr}
              </pre>
            ) : null}
          </div>
        );
      },
    },
    [],
  );

  return null;
}
