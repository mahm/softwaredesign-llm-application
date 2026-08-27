import { OpenRouter } from "@openrouter/sdk";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  throw new Error("OPENROUTER_API_KEY is required");
}

const openRouter = new OpenRouter({ apiKey });
let generation = await openRouter.videoGeneration.generate({
  videoGenerationRequest: {
    model: "bytedance/seedance-2.0-fast",
    prompt:
      "A small paper robot walks across a desk and opens a glowing book, locked camera, soft studio lighting",
    aspectRatio: "1:1",
    duration: 4,
    resolution: "480p",
    generateAudio: false,
  },
});

console.log(`Video job ${generation.id}: ${generation.status}`);
while (generation.status === "pending" || generation.status === "in_progress") {
  await Bun.sleep(5_000);
  generation = await openRouter.videoGeneration.getGeneration({
    jobId: generation.id,
  });
  console.log(`Video job ${generation.id}: ${generation.status}`);
}

if (generation.status !== "completed") {
  throw new Error(
    `Video generation ${generation.status}: ${generation.error ?? "unknown error"}`,
  );
}

const video = await openRouter.videoGeneration.getVideoContent({
  jobId: generation.id,
  index: 0,
});
const outputDirectory = new URL("../outputs/", import.meta.url);
await mkdir(outputDirectory, { recursive: true });
const outputPath = new URL("video.mp4", outputDirectory);
await Bun.write(outputPath, Buffer.concat(await Array.fromAsync(video)));

console.log(`Saved video to ${fileURLToPath(outputPath)}`);
