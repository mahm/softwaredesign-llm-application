# Terminal-Bench Harness Tuning Prompt

You are tuning the Deep Agents harness in `35/`.

The harness is a JSON file passed to `createAgent({ harnessFile })`.
Harbor is only the evaluation wrapper.

## Goal

Create a small improvement sequence for the article:

```text
baseline -> improvement-1 -> improvement-2
```

This sequence is a one-attempt `test10` comparison.

## Hard Rules

Edit only new `harness-runs/<name>/harness.json` files.

Do not edit:

- `src/`
- `adapters/`
- `scripts/`
- `smoke_tasks/`
- scoring or reporting code

Do not put task names, task-specific file paths, known answers, or task-specific command sequences into `systemPrompt`.

Use traces only to extract general agent behavior failures, such as:

- missing required output files
- stopping before verification
- repeating a failed strategy
- spending too long on unproductive exploration
- ignoring the requested output format

Convert those observations into generic prompt guidance or numeric harness settings.

## Data Splits

Use `dev5` for tuning.
You may inspect `dev5` traces.
`dev5` runs use 3 attempts by default.

Use `test10` for one-attempt screening only.
Do not inspect `test10` traces or per-trial logs while tuning.
Read only `result.json` and `suite-summary.json`.
`test10` is fixed to `N_ATTEMPTS=1`.

## Workflow

1. Run the baseline on `dev5`.
2. Run the baseline on `test10`.
3. Inspect only the `dev5` traces.
4. Create `harness-runs/improvement-1/harness.json`.
5. Run `improvement-1` on `dev5` and `test10`.
6. Inspect only the `dev5` traces.
7. Create `harness-runs/improvement-2/harness.json`.
8. Run `improvement-2` on `dev5` and `test10`.
9. Write a short `test10` k=1 comparison table.

## Commands

Run `dev5` with trace collection:

```bash
HARNESS_FILE=harness-runs/baseline/harness.json \
  CONCURRENCY=1 USE_SHARED_DOCKER_BRIDGE=1 \
  bun run harbor:dev5 -- dev5-baseline deepseek/deepseek-v4-flash
```

Run `test10` as a one-attempt screening run:

```bash
HARNESS_FILE=harness-runs/baseline/harness.json \
  CONCURRENCY=1 USE_SHARED_DOCKER_BRIDGE=1 \
  bun run harbor:test10 -- t10s1-baseline deepseek/deepseek-v4-flash
```

Use the same command shape for `improvement-1` and `improvement-2`.

## Acceptance

Prefer higher `trialPasses` on `test10` k=1.
For k=1, `strictTaskPasses` is expected to match `trialPasses`.

Use `exceptionCount` as a caution signal, not as the primary score.
If a candidate improves pass count but increases timeout exceptions, keep the pass improvement and record the timeout caveat.

Reject a candidate when it has the same pass count as another candidate and more exceptions.

Do not make stability claims from this comparison.

## Records

Keep only score artifacts needed for the article comparison:

- `result.json`
- `suite-summary.json`
- a short comparison table

Do not keep detailed `test10` traces after summarizing the score.
