import { OpenRouter } from "@openrouter/sdk";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  throw new Error("OPENROUTER_API_KEY is required");
}

const openRouter = new OpenRouter({ apiKey });
const speech = await openRouter.tts.createSpeech({
  speechRequest: {
    model: "qwen/qwen-audio-3.0-tts-flash",
    input:
      "OpenRouterを使うと、さまざまな開発元のモデルを一つのAPIから利用できます。",
    voice: "loongjohn",
    responseFormat: "mp3",
  },
});

const outputDirectory = new URL("../outputs/", import.meta.url);
await mkdir(outputDirectory, { recursive: true });
const outputPath = new URL("speech.mp3", outputDirectory);
await Bun.write(outputPath, Buffer.concat(await Array.fromAsync(speech)));

console.log(`Saved speech to ${fileURLToPath(outputPath)}`);
