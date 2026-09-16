import { z } from "zod";

const model = "deepseek/deepseek-v4-flash-0731";
const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
const file = Bun.file(new URL("../fixtures/pdf/quote-scan.pdf", import.meta.url));
const fileData = `data:application/pdf;base64,${Buffer.from(await file.arrayBuffer()).toString("base64")}`;
const firstMessage = { role: "user", content: [
  { type: "text", text: "Read the quote pages. Return JSON with quoteId, totalYen, and validUntil (YYYY-MM-DD). Use exact values from the page contents." },
  { type: "file", file: { filename: "quote-scan.pdf", file_data: fileData } },
] };
const initialSchema = z.object({ quoteId: z.literal("SD38-042"), totalYen: z.literal(2240), validUntil: z.literal("2026-10-31") }).strict();
const followupSchema = z.object({ destination: z.literal("Meeting Room B"), referenceCode: z.literal("COBALT-73") }).strict();

async function send(messages: unknown[]) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST", headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json", "X-OpenRouter-Metadata": "enabled", "X-OpenRouter-Cache": "false" },
    body: JSON.stringify({ model, messages, provider: { only: ["deepinfra"], allow_fallbacks: false },
      reasoning_effort: "none", temperature: 0, max_tokens: 256, response_format: { type: "json_object" },
      plugins: [{ id: "file-parser", pdf: { engine: "mistral-ocr" } }], stream: false }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const result = await response.json();
  if (result.error || result.choices?.[0]?.finish_reason !== "stop") throw new Error("The model did not complete its answer");
  return result;
}

try {
  const first = await send([firstMessage]);
  const assistant = first.choices[0].message;
  initialSchema.parse(JSON.parse(assistant.content));
  if (!assistant.annotations?.some((a: { type?: string }) => a.type === "file")) throw new Error("No parsed PDF annotation returned");
  const question = { role: "user", content: "Return JSON with destination and referenceCode from page 2. Use exact values from the page contents." };
  const second = await send([firstMessage, { role: "assistant", content: assistant.content, annotations: assistant.annotations }, question]);
  const answer = followupSchema.parse(JSON.parse(second.choices[0].message.content));
  console.log(JSON.stringify({ model, firstCost: first.usage?.cost, followupCost: second.usage?.cost,
    answer, firstPipeline: first.openrouter_metadata?.pipeline ?? [], followupPipeline: second.openrouter_metadata?.pipeline ?? [] }, null, 2));
  if (process.argv.includes("--compare")) {
    const without = await send([firstMessage, { role: "assistant", content: assistant.content }, question]);
    followupSchema.parse(JSON.parse(without.choices[0].message.content));
    console.log(JSON.stringify({ withoutAnnotationsCost: without.usage?.cost, pipeline: without.openrouter_metadata?.pipeline ?? [] }, null, 2));
  }
} catch (error) {
  console.error(JSON.stringify({ error: (error as Error).name }));
  process.exitCode = 1;
}
