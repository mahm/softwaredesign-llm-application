import { OpenRouter } from "@openrouter/sdk";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  throw new Error("OPENROUTER_API_KEY is required");
}

const openRouter = new OpenRouter({ apiKey });
const result = await openRouter.images.generate({
  imageGenerationRequest: {
    model: "krea/krea-2-medium",
    prompt:
      "A conceptual editorial illustration of OpenRouter as a central AI gateway: one application sends a glowing request into an open routing hub, which branches to several distinct AI model nodes and returns one response, clear left-to-right flow, modern technology magazine style, dark background with vivid cyan, violet, and coral accents, no logos, no readable text",
    aspectRatio: "1:1",
    resolution: "1K",
    stream: false,
  },
});

if (!("data" in result)) {
  throw new Error("OpenRouter returned an unexpected streaming response");
}

const image = result.data[0];
if (!image) {
  throw new Error("The model did not return an image");
}

const outputDirectory = new URL("../outputs/", import.meta.url);
await mkdir(outputDirectory, { recursive: true });
const outputPath = new URL("image.png", outputDirectory);
await Bun.write(outputPath, Buffer.from(image.b64Json, "base64"));

console.log(`Saved image to ${fileURLToPath(outputPath)}`);
