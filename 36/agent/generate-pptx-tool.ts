import { tool } from "@langchain/core/tools";
import type { BackendProtocolV2 } from "deepagents";
import { z } from "zod";

const slideSchema = z.object({
  type: z.enum(["title", "section", "content"]),
  title: z.string(),
  subtitle: z.string().optional(),
  bullets: z.array(z.string()).optional(),
});

const slideDataSchema = z.object({
  title: z.string(),
  author: z.string().optional(),
  slides: z.array(slideSchema),
});

const inputSchema = z.object({
  filePath: z
    .string()
    .describe("スライドJSONファイルのパス(例: ./slides/2603.03303.json)"),
});

export function createGeneratePptxTool(backend: BackendProtocolV2) {
  return tool(
    async (input) => {
      const readResult = await backend.readRaw(input.filePath);
      const content = readResult.data?.content;
      if (readResult.error || content === undefined) {
        return JSON.stringify({
          success: false,
          error: "file_not_found",
          details: `ファイルが見つかりません: ${input.filePath}`,
          action: "ファイルパスを確認し、正しいパスで再度呼び出してください。",
        });
      }
      const raw =
        typeof content === "string"
          ? content
          : Array.isArray(content)
            ? content.join("\n")
            : new TextDecoder().decode(content);

      let json: unknown;
      try {
        json = JSON.parse(raw);
      } catch (e) {
        return JSON.stringify({
          success: false,
          error: "invalid_json",
          details: `JSONパースエラー: ${e instanceof Error ? e.message : String(e)}`,
          action: `${input.filePath} のJSON構文を修正し、再度このツールを呼び出してください。`,
        });
      }

      const result = slideDataSchema.safeParse(json);
      if (!result.success) {
        const details = result.error.issues.map((issue) => ({
          path: issue.path.join("."),
          message: issue.message,
        }));
        return JSON.stringify({
          success: false,
          error: "validation",
          details,
          action: `${input.filePath} を以下のエラー内容に基づいて修正し、再度このツールを呼び出してください。`,
        });
      }

      const data = result.data;
      return JSON.stringify({
        success: true,
        title: data.title,
        author: data.author,
        slideCount: data.slides.length,
        slides: data.slides,
      });
    },
    {
      name: "generate_pptx",
      description:
        "スライドJSONファイルを読み込み、スキーマ検証してPowerPointファイル(.pptx)を生成する。スライドのアウトラインが確定した後に呼び出す。",
      schema: inputSchema,
    },
  );
}
