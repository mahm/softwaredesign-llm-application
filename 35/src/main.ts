import { config as loadDotenv } from "dotenv";

import {
  DEFAULT_HARNESS_FILE,
  reportRun,
  runHarborTask,
} from "./harbor-runner.js";

loadDotenv({ path: new URL("../.env", import.meta.url).pathname, quiet: true });

const command = process.argv[2];

if (command === "run") {
  await runHarborTask(
    process.argv[3] ?? "/tmp/task.md",
    process.argv[4] ?? "results/manual",
    process.argv[5] ?? DEFAULT_HARNESS_FILE,
    process.argv[6] ?? "full"
  );
} else if (command === "report") {
  reportRun(process.argv[3] ?? "results/manual");
} else {
  console.error("Usage: bun src/main.ts <run|report>");
  process.exit(1);
}
