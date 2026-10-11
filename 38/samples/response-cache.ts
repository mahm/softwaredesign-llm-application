import { OpenRouter } from "@openrouter/sdk";
import { HTTPClient } from "@openrouter/sdk/lib/http.js";
import type { ChatRequest } from "@openrouter/sdk/models";
import assert from "node:assert/strict";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");

const httpClient = new HTTPClient();
let cacheStatus: string | null = null;
let sourceId: string | null = null;
httpClient.addHook("response", (response) => {
  cacheStatus = response.headers.get("x-openrouter-cache-status");
  sourceId = response.headers.get("x-openrouter-cache-source-id");
});
const client = new OpenRouter({ apiKey, httpClient, retryConfig: { strategy: "none" } });

// 実行ごとに識別子を変え、同じ実行内では本文を変えずに再送します。
const request: ChatRequest = {
  model: "deepseek/deepseek-v4-flash-0731",
  provider: { only: ["deepinfra/fp8"], allowFallbacks: false },
  messages: [{ role: "user", content: `Run ${crypto.randomUUID()}. Reply with exactly CACHE-OK.` }],
  reasoning: { effort: "none" },
  maxTokens: 64,
  temperature: 0,
  stream: false,
};

const results = [];
for (let attempt = 1; attempt <= 2; attempt++) {
  const completion = await client.chat.send({ chatRequest: request }, {
    headers: { "X-OpenRouter-Cache": "true", "X-OpenRouter-Cache-TTL": "120" },
    retries: { strategy: "none" },
    signal: AbortSignal.timeout(60_000),
  });
  if (Symbol.asyncIterator in completion) throw new Error("Expected a non-streaming response");
  assert.equal(completion.choices[0]?.finishReason, "stop");
  results.push({ attempt, cacheStatus, sourceId, id: completion.id, model: completion.model,
    text: completion.choices[0]?.message.content, usage: completion.usage });
  console.log(JSON.stringify(results.at(-1), null, 2));
  if (attempt === 1) await Bun.sleep(1500);
}
