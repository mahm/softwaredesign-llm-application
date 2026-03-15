"use client";

import { useSlideData } from "./slide-context";

function TitleSlide({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="flex aspect-video flex-col justify-center rounded-lg bg-[#1B2A4A] p-6">
      <h3 className="text-lg font-bold text-white">{title}</h3>
      {subtitle && <p className="mt-2 text-sm text-slate-300">{subtitle}</p>}
    </div>
  );
}

function SectionSlide({ title }: { title: string }) {
  return (
    <div className="flex aspect-video flex-col justify-center rounded-lg bg-[#2E86AB] p-6">
      <h3 className="text-lg font-bold text-white">{title}</h3>
    </div>
  );
}

function ContentSlide({
  title,
  bullets,
}: { title: string; bullets?: string[] }) {
  return (
    <div className="flex aspect-video flex-col rounded-lg border border-slate-200 bg-white p-6">
      <h3 className="text-base font-bold text-[#1B2A4A]">{title}</h3>
      <div className="mt-1 h-0.5 w-full bg-[#2E86AB]" />
      {bullets && bullets.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {bullets.map((bullet) => (
            <li
              key={`${title}-${bullet.slice(0, 20)}`}
              className="flex items-start gap-2 text-sm text-slate-700"
            >
              <span className="mt-1.5 inline-block size-1.5 shrink-0 rounded-full bg-[#2E86AB]" />
              {bullet}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SlidePreview() {
  const { slideData } = useSlideData();

  if (!slideData) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-slate-400">
          論文URLをチャットに入力すると、ここにスライドのプレビューが表示されます
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-sm font-medium text-slate-600">
        {slideData.title} ({slideData.slides.length} slides)
      </h2>
      <div className="grid gap-4">
        {slideData.slides.map((slide, i) => (
          <div key={`slide-${slide.type}-${i}`}>
            <span className="mb-1 block text-xs text-slate-400">
              Slide {i + 1}
            </span>
            {slide.type === "title" ? (
              <TitleSlide title={slide.title} subtitle={slide.subtitle} />
            ) : slide.type === "section" ? (
              <SectionSlide title={slide.title} />
            ) : (
              <ContentSlide title={slide.title} bullets={slide.bullets} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
