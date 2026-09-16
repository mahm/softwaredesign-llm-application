import { OpenRouter } from "@openrouter/sdk";
import { z } from "zod";

const model = "deepseek/deepseek-v4-flash-0731";
const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
const client = new OpenRouter({ apiKey, retryConfig: { strategy: "none" } });
const productSchema = z.object({
  name: z.literal("Notebook"), unitPrice: z.literal(480), quantity: z.literal(3),
}).strict();
// 修復の実演では、stopで閉じ括弧を意図的に取り除きます。
// --normalを指定すると、この実演用のstopを使わず1回実行します。
const demonstrateMissingBrace = !process.argv.includes("--normal");

try {
  for (const enabled of demonstrateMissingBrace ? [false, true] : [true]) {
    const result = await client.chat.send({ xOpenRouterMetadata: "enabled", chatRequest: {
      model,
      messages: [{ role: "user", content: "Extract this fictional product memo as one JSON object with name (string), unitPrice (number), and quantity (integer): Product: Notebook. Unit price: 480 yen. Quantity: 3." }],
      provider: { only: ["deepinfra"], allowFallbacks: false },
      reasoningEffort: "none", temperature: 0, maxTokens: 192,
      responseFormat: { type: "json_object" },
      plugins: [{ id: "response-healing", enabled }],
      ...(demonstrateMissingBrace ? { stop: ["}"] } : {}),
      stream: false,
    } }, { headers: { "X-OpenRouter-Cache": "false" }, retries: { strategy: "none" }, signal: AbortSignal.timeout(60_000) });
    if (!("choices" in result)) throw new Error("Expected a non-streaming response");
    const choice = result.choices[0];
    if (!choice || choice.finishReason !== "stop" || typeof choice.message.content !== "string") throw new Error("Expected a stop response with text");
    let parsed: unknown;
    let jsonValid = false;
    try { parsed = JSON.parse(choice.message.content); jsonValid = true; } catch {}
    const schemaValid = productSchema.safeParse(parsed).success;
    if (enabled && !schemaValid) throw new Error("The response does not match the expected product");
    console.log(JSON.stringify({ model: result.model, enabled, demonstrateMissingBrace,
      finishReason: choice.finishReason, content: choice.message.content,
      jsonValid, schemaValid, usage: result.usage }, null, 2));
  }
} catch (error) {
  console.error(JSON.stringify({ error: (error as Error).name }));
  process.exitCode = 1;
}
