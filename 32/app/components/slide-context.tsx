"use client";

import { type ReactNode, createContext, useContext, useState } from "react";

export interface SlideData {
  title: string;
  author?: string;
  slides: Array<{
    type: "title" | "section" | "content";
    title: string;
    subtitle?: string;
    bullets?: string[];
  }>;
}

interface SlideContextValue {
  slideData: SlideData | null;
  pptxBase64: string | null;
  setSlideData: (data: SlideData) => void;
  setPptxBase64: (base64: string) => void;
}

const SlideContext = createContext<SlideContextValue>({
  slideData: null,
  pptxBase64: null,
  setSlideData: () => {},
  setPptxBase64: () => {},
});

export function SlideProvider({ children }: { children: ReactNode }) {
  const [slideData, setSlideData] = useState<SlideData | null>(null);
  const [pptxBase64, setPptxBase64] = useState<string | null>(null);

  return (
    <SlideContext
      value={{ slideData, pptxBase64, setSlideData, setPptxBase64 }}
    >
      {children}
    </SlideContext>
  );
}

export function useSlideData() {
  return useContext(SlideContext);
}
