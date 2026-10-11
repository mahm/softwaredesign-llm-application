import { OpenRouter } from "@openrouter/sdk";
import { HTTPClient } from "@openrouter/sdk/lib/http.js";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey || apiKey.startsWith("op://"))
  throw new Error("A resolved OPENROUTER_API_KEY is required");

export async function companyNews(company: string, ttlSeconds = 900) {
  const name = company.trim();
  if (!name) throw new Error("Company name is required");
  const today = new Date().toLocaleDateString("sv-SE", { timeZone: "Asia/Tokyo" });
  const httpClient = new HTTPClient();
  let headers = new Headers();
  httpClient.addHook("response", response => { headers = response.headers; });
  const client = new OpenRouter({ apiKey, httpClient });
  const started = performance.now();
  const result = await client.chat.send({ chatRequest: {
    model: "deepseek/deepseek-v4-flash-0731",
    provider: { only: ["deepinfra/fp8"], allowFallbacks: false },
    messages: [
      { role: "system", content:
        `今日は${today}です。指定された企業の最新ニュースをWeb検索で調べてください。` +
        "確認できたニュースを最大3件、日本語で簡潔に箇条書きにしてください。" +
        "各項目には発表日または公開日、要点、出典記事URLを含め、確認できない事実は補わないでください。" },
      { role: "user", content: `${name}の最新ニュースを調べてください。` },
    ],
    tools: [{ type: "openrouter:web_search",
      parameters: { engine: "exa", maxResults: 5, maxUses: 1 } }],
    stopServerToolsWhen: [{ type: "step_count_is", stepCount: 1 }],
    reasoning: { effort: "none" },
    maxTokens: 1200, temperature: 0, stream: false,
  } }, {
    headers: { "X-OpenRouter-Cache": "true",
      "X-OpenRouter-Cache-TTL": String(ttlSeconds) },
    signal: AbortSignal.timeout(90_000), retries: { strategy: "none" },
  });
  if (!("choices" in result)) throw new Error("Expected JSON response");
  const choice = result.choices[0];
  if (choice?.finishReason !== "stop" || typeof choice.message.content !== "string")
    throw new Error("Expected complete text");
  return { text: choice.message.content,
    elapsedSeconds: (performance.now() - started) / 1000,
    cacheStatus: headers.get("x-openrouter-cache-status"),
    cacheAge: headers.get("x-openrouter-cache-age"),
    usage: result.usage };
}

if (import.meta.main) {
  try {
    const company = process.argv[2] ?? "ソニーグループ";
    console.log(await companyNews(company));
    await Bun.sleep(1500);
    console.log(await companyNews(company));
  } catch (error) {
    console.error(error instanceof Error ? error.name : "UnknownError");
    process.exitCode = 1;
  }
}
