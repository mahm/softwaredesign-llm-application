import { tool } from "@langchain/core/tools";
import { z } from "zod";

const slideSchema = z.object({
  type: z.enum(["title", "section", "content"]),
  title: z.string(),
  subtitle: z.string().optional(),
  bullets: z.array(z.string()).optional(),
});

const inputSchema = z.object({
  title: z.string().describe("プレゼンテーションのタイトル"),
  author: z.string().optional().describe("著者名"),
  slides: z.array(slideSchema).describe("スライドデータの配列"),
});

export const generatePptxTool = tool(
  async (input) => {
    return JSON.stringify({
      success: true,
      title: input.title,
      author: input.author,
      slideCount: input.slides.length,
    });
  },
  {
    name: "generate_pptx",
    description:
      "スライドデータからPowerPointファイル(.pptx)を生成する。スライドのアウトラインが確定した後に呼び出す。",
    schema: inputSchema,
  },
);
