import fs from "node:fs";
import path from "node:path";

type TrialResult = {
  trial_name?: string;
  task_name?: string;
  verifier_result?: {
    rewards?: {
      reward?: number;
    };
  };
  exception_info?: {
    exception_type?: string;
  } | null;
};

type AgentSummary = {
  toolCalls?: number;
  failedToolCalls?: number;
  editCalls?: number;
  executeCalls?: number;
};

type CheckpointSnapshot = {
  checkpointCount?: number;
};

type TrialSummary = {
  trialName: string;
  reward: number;
  exceptionType: string | null;
  failedToolCalls: number | null;
  toolCalls: number | null;
  checkpointCount: number | null;
};

type TaskSummary = {
  taskName: string;
  attempts: number;
  passes: number;
  strictPass: boolean;
  trials: TrialSummary[];
};

type SuiteSummary = {
  jobDir: string;
  expectedAttempts: number;
  trialCount: number;
  trialPasses: number;
  trialMean: number;
  strictTaskPasses: number;
  taskCount: number;
  strictTaskMean: number;
  exceptionCount: number;
  failedToolCalls: number;
  tasks: TaskSummary[];
};

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
}

function maybeReadJson<T>(file: string): T | null {
  if (!fs.existsSync(file)) {
    return null;
  }

  return readJson<T>(file);
}

function writeJson(file: string, value: unknown) {
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function formatNumber(value: number): string {
  return value.toFixed(3);
}

function taskSortKey(taskName: string): string {
  return taskName.split("/").at(-1) ?? taskName;
}

function loadTrialSummary(trialDir: string): { taskName: string; trial: TrialSummary } {
  const result = readJson<TrialResult>(path.join(trialDir, "result.json"));
  const taskName = result.task_name ?? path.basename(trialDir).split("__")[0] ?? "unknown";
  const agentSummary = maybeReadJson<AgentSummary>(
    path.join(trialDir, "agent", "deepagents-ts", "summary.json")
  );
  const checkpoints = maybeReadJson<CheckpointSnapshot>(
    path.join(trialDir, "agent", "deepagents-ts", "checkpoints.json")
  );

  return {
    taskName,
    trial: {
      trialName: result.trial_name ?? path.basename(trialDir),
      reward: Number(result.verifier_result?.rewards?.reward ?? 0),
      exceptionType: result.exception_info?.exception_type ?? null,
      failedToolCalls: agentSummary?.failedToolCalls ?? null,
      toolCalls: agentSummary?.toolCalls ?? null,
      checkpointCount: checkpoints?.checkpointCount ?? null,
    },
  };
}

function buildSuiteSummary(jobDir: string, expectedAttempts: number): SuiteSummary {
  const taskMap = new Map<string, TrialSummary[]>();

  for (const entry of fs.readdirSync(jobDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) {
      continue;
    }

    const trialDir = path.join(jobDir, entry.name);
    const resultFile = path.join(trialDir, "result.json");
    if (!fs.existsSync(resultFile)) {
      continue;
    }

    const { taskName, trial } = loadTrialSummary(trialDir);
    const trials = taskMap.get(taskName) ?? [];
    trials.push(trial);
    taskMap.set(taskName, trials);
  }

  const tasks: TaskSummary[] = [...taskMap.entries()]
    .sort(([left], [right]) => taskSortKey(left).localeCompare(taskSortKey(right)))
    .map(([taskName, trials]) => {
      const sortedTrials = [...trials].sort((left, right) =>
        left.trialName.localeCompare(right.trialName)
      );
      const passes = sortedTrials.filter((trial) => trial.reward === 1).length;

      return {
        taskName,
        attempts: sortedTrials.length,
        passes,
        strictPass:
          sortedTrials.length === expectedAttempts &&
          sortedTrials.every((trial) => trial.reward === 1),
        trials: sortedTrials,
      };
    });

  const allTrials = tasks.flatMap((task) => task.trials);
  const trialPasses = allTrials.filter((trial) => trial.reward === 1).length;
  const strictTaskPasses = tasks.filter((task) => task.strictPass).length;
  const exceptionCount = allTrials.filter((trial) => trial.exceptionType).length;
  const failedToolCalls = allTrials.reduce(
    (total, trial) => total + (trial.failedToolCalls ?? 0),
    0
  );

  return {
    jobDir,
    expectedAttempts,
    trialCount: allTrials.length,
    trialPasses,
    trialMean: allTrials.length === 0 ? 0 : trialPasses / allTrials.length,
    strictTaskPasses,
    taskCount: tasks.length,
    strictTaskMean: tasks.length === 0 ? 0 : strictTaskPasses / tasks.length,
    exceptionCount,
    failedToolCalls,
    tasks,
  };
}

function printSummary(summary: SuiteSummary) {
  console.log(`job: ${summary.jobDir}`);
  console.log(
    `trial mean: ${summary.trialPasses}/${summary.trialCount} = ${formatNumber(
      summary.trialMean
    )}`
  );
  console.log(
    `strict task mean (${summary.expectedAttempts}/${summary.expectedAttempts}): ` +
      `${summary.strictTaskPasses}/${summary.taskCount} = ${formatNumber(
        summary.strictTaskMean
      )}`
  );
  console.log(`exceptions: ${summary.exceptionCount}`);
  console.log(`failed tool calls: ${summary.failedToolCalls}`);
  console.log("");

  for (const task of summary.tasks) {
    const mark = task.strictPass ? "PASS" : "FAIL";
    console.log(`${mark} ${task.taskName}: ${task.passes}/${task.attempts}`);
  }
}

const [jobDirArg, expectedAttemptsArg] = process.argv.slice(2);

if (!jobDirArg) {
  console.error("Usage: bun src/harbor-report.ts <job-dir> [expected-attempts]");
  process.exit(2);
}

const jobDir = path.resolve(jobDirArg);
const expectedAttempts = Number(expectedAttemptsArg ?? process.env.N_ATTEMPTS ?? 3);

if (!Number.isInteger(expectedAttempts) || expectedAttempts < 1) {
  console.error("expected-attempts must be a positive integer");
  process.exit(2);
}

const summary = buildSuiteSummary(jobDir, expectedAttempts);
writeJson(path.join(jobDir, "suite-summary.json"), summary);
printSummary(summary);
