import { OpenRouter } from "@openrouter/sdk";
import { z } from "zod";

const model = "deepseek/deepseek-v4-flash-0731";
const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
const engine = z.enum(["cloudflare-ai", "mistral-ocr", "native"]).parse(process.argv[2] ?? "cloudflare-ai");
const file = Bun.file(process.argv[3] ?? new URL("../fixtures/pdf/quote-text.pdf", import.meta.url));
const quoteSchema = z.object({
  quoteId: z.string(), totalYen: z.number(), validUntil: z.string(),
  destination: z.string(), referenceCode: z.string(),
}).strict();
const expected = { quoteId: "SD38-042", totalYen: 2240, validUntil: "2026-10-31", destination: "Meeting Room B", referenceCode: "COBALT-73" };
const client = new OpenRouter({ apiKey, retryConfig: { strategy: "none" } });

try {
  const result = await client.chat.send({ xOpenRouterMetadata: "enabled", chatRequest: {
    model,
    messages: [{ role: "user", content: [
      { type: "text", text: "Read the quote pages and return JSON with quoteId, totalYen, validUntil (YYYY-MM-DD), destination, and referenceCode. Use exact values from the page contents. Do not infer values from the file name or PDF metadata. Use null for information missing from the pages." },
      { type: "file", file: { filename: file.name?.split("/").at(-1) ?? "quote.pdf", fileData: `data:application/pdf;base64,${Buffer.from(await file.arrayBuffer()).toString("base64")}` } },
    ] }],
    provider: { only: ["deepinfra"], allowFallbacks: false },
    reasoningEffort: "none", temperature: 0, maxTokens: 256,
    responseFormat: { type: "json_object" },
    plugins: [{ id: "file-parser", pdf: { engine } }], stream: false,
  } }, { headers: { "X-OpenRouter-Cache": "false" }, retries: { strategy: "none" }, signal: AbortSignal.timeout(60_000) });
  if (!("choices" in result)) throw new Error("Expected a non-streaming response");
  const choice = result.choices[0];
  if (!choice || choice.finishReason !== "stop" || typeof choice.message.content !== "string") throw new Error("The model did not complete its answer");
  const quote = quoteSchema.parse(JSON.parse(choice.message.content));
  const matchesFixture = Object.entries(expected).every(([name, value]) => quote[name as keyof typeof quote] === value);
  if (!matchesFixture) throw new Error("The extracted values do not match the fixture");
  console.log(JSON.stringify({ model: result.model, engine, quote, matchesFixture, usage: result.usage }, null, 2));
} catch (error) {
  // SDK例外全体を表示せず、認証情報を含まない要約を返します。
  const status = (error as { statusCode?: number }).statusCode;
  console.error(JSON.stringify({ error: (error as Error).name, ...(status ? { status } : {}) }));
  process.exitCode = 1;
}
