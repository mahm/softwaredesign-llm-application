import { OpenRouter } from "@openrouter/sdk";
import type { ChatRequest } from "@openrouter/sdk/models";
import { HTTPClient } from "@openrouter/sdk/lib/http.js";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
const mode = process.argv[2] ?? "short";
if (!["short", "off", "on"].includes(mode)) throw new Error("Use short, off, or on");
const messages: ChatRequest["messages"] = [
  { role: "user", content: "Project code: ALPHA-173." },
];
for (let i = 0; i < 29; i++) {
  messages.push({ role: i % 2 ? "user" : "assistant",
    content: (i === 15 ? "Order quantity: 47.\n" : "") +
      "Background: " + "apple ".repeat(mode === "short" ? 20 : 30_000) });
}
messages.push({ role: "user", content: "Delivery room: ROOM-B. " +
  "Read ALL preceding user and assistant messages. Extract project code, order quantity, " +
  "and delivery room. Return JSON with project, quantity, room. Use null for absent values." });

// The SDK drops debug events. Capture a clone without consuming its response.
let rawResponse: Promise<string> = Promise.resolve("");
const httpClient = new HTTPClient();
httpClient.addHook("response", response => { rawResponse = response.clone().text(); });
const openRouter = new OpenRouter({ apiKey, httpClient });
const stream = await openRouter.chat.send({ chatRequest: {
  model: "deepseek/deepseek-v4-flash-0731", messages,
  provider: { only: ["deepinfra"], allowFallbacks: false },
  plugins: [{ id: "context-compression", enabled: mode !== "off", engine: "middle-out" }],
  maxTokens: 128, reasoningEffort: "none", temperature: 0,
  stream: true, debug: { echoUpstreamBody: true },
} , xOpenRouterMetadata: "enabled" }, { headers: { "X-OpenRouter-Cache": "false" },
  retries: { strategy: "none" }, signal: AbortSignal.timeout(mode === "short" ? 60_000 : 180_000) });
if (!(Symbol.asyncIterator in stream)) throw new Error("Expected stream");
let text = "", finish: string | null | undefined;
for await (const chunk of stream) {
  text += chunk.choices[0]?.delta.content ?? "";
  finish = chunk.choices[0]?.finishReason ?? finish;
  if (chunk.usage) console.log(JSON.stringify({ usage: chunk.usage }));
}
if (finish !== "stop" || !text) throw new Error("Incomplete response");
const events = (await rawResponse).split(/\r?\n/)
  .filter(line => line.startsWith("data:")).map(line => line.slice(5).trim())
  .filter(line => line && line !== "[DONE]").map(line => JSON.parse(line));
const debug = events.find(event => event.debug)?.debug;
const upstream = debug?.echo_upstream_body;
const sent = upstream ? JSON.stringify(upstream) : undefined;
const metadata = events.findLast(event => event.openrouter_metadata)?.openrouter_metadata;
console.log(JSON.stringify({ mode, inputMessages: messages.length,
  sentMessages: upstream?.messages?.length, sentCharacters: sent?.length,
  hasQuantity: sent ? sent.includes("Order quantity: 47") : null,
  pipeline: metadata?.pipeline, text, finish }));
