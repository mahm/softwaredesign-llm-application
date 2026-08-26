import { OpenRouter } from "@openrouter/sdk";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  throw new Error("OPENROUTER_API_KEY is required");
}

const openRouter = new OpenRouter({ apiKey });
const result = await openRouter.chat.send({
  chatRequest: {
    model: "deepseek/deepseek-v4-flash-0731",
    messages: [
      {
        role: "user",
        content: "OpenRouterを一文で説明してください。",
      },
    ],
    maxCompletionTokens: 200,
    reasoningEffort: "none",
    stream: false,
  },
});

if (!("choices" in result)) {
  throw new Error("OpenRouter returned an unexpected streaming response");
}

const text = result.choices[0]?.message.content as string;

const outputDirectory = new URL("../outputs/", import.meta.url);
await mkdir(outputDirectory, { recursive: true });
const outputPath = new URL("text.txt", outputDirectory);
await Bun.write(outputPath, text);

console.log(`Saved text to ${fileURLToPath(outputPath)}`);
