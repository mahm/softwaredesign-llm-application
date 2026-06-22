# Software Design誌「実践LLMアプリケーション開発」第35回サンプルコード

DeepAgents TypeScript と OpenRouter を使って、Harbor 経由で Terminal-Bench 2 の公式タスクを実行するサンプルです。
このサンプルでは、人間が `harness.json` を手で作り込むのではなく、`PROMPT.md` を読ませたコーディングエージェントに改善候補を作らせ、その候補を Harbor で測定します。

## 前提条件

- [mise](https://mise.jdx.dev/)
- Bun
- Docker
- `uv` / `uvx`
- [Harbor](https://www.harborframework.com/docs/tutorials/running-terminal-bench)
- OpenRouter APIキー

## セットアップ

```bash
mise trust 35/.mise.toml
cd 35
mise install
mise run install
uv tool install harbor
```

`.env` を作成し、OpenRouter APIキーを設定します。

```bash
cp .env.sample .env
vi .env
```

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## ハーネス設定

実行条件は `harness-runs/<name>/harness.json` に記述します。

```json
{
  "model": "deepseek/deepseek-v4-flash",
  "temperature": 1,
  "recursionLimit": 150,
  "commandTimeoutSec": 180,
  "maxOutputBytes": 160000,
  "systemPrompt": [
    "You are an autonomous coding agent working inside a terminal task."
  ]
}
```

主な項目です。

- `model`: OpenRouter のモデルID
- `temperature`: モデル呼び出し時の temperature
- `recursionLimit`: DeepAgents / LangGraph の最大ステップ数
- `commandTimeoutSec`: shell command 1回あたりの timeout 秒数
- `agentRunTimeoutSec`: agent runner 全体の timeout 秒数
- `maxOutputBytes`: shell command 1回あたりの出力上限
- `systemPrompt`: エージェントに渡す system prompt

`bun run harbor:test10 -- <label> <model>` でモデルIDを指定した場合は、その値を Harbor に `BASE_MODEL` として渡し、`harness.json` の `model` より優先します。
`dev5` のように trace を有効にした実行では、実際に使った値が各 trial の `harness.used.json` と `run.json` に保存されます。

`harness.json` は Harbor 専用ではなく、`agent.ts` が直接読み込むエージェント実行設定です。
最適化済みの harness は、ベンチマークと切り離して単体実行できます。

```bash
printf 'Create /tmp/answer.txt with the text hello' \
  | bun src/agent.ts run - harness-runs/baseline/harness.json results/manual-agent

bun src/agent.ts run /tmp/task.md harness-runs/improvement-1/harness.json results/manual-improvement-1
```

`agent.ts` の主な API は次の通りです。

- `createAgent({ harnessFile })`: `harness.json` を読み込み、実行可能なエージェントインスタンスを作る
- `createAgent({ harness })`: 読み込み済みの harness 設定からエージェントインスタンスを作る
- `runAgent(agent, task)`: 作成済みのエージェントインスタンスでタスクを実行する

## 公式Terminal-Bench 2を実行する

`test10` では、`baseline -> improvement-1 -> improvement-2` の3本を比較します。
`test10` は測定専用タスクで、ハーネス調整時に trace を読まない前提のセットです。
実行時間を抑えるため、`harbor:test10` は `N_ATTEMPTS=1` に固定しています。
この結果は1回試行の比較です。

比較条件をそろえる場合は、`CONCURRENCY=1` と `USE_SHARED_DOCKER_BRIDGE=1` を指定します。

```bash
HARNESS_FILE=harness-runs/baseline/harness.json \
  CONCURRENCY=1 USE_SHARED_DOCKER_BRIDGE=1 \
  bun run harbor:test10 -- t10s1-baseline deepseek/deepseek-v4-flash

HARNESS_FILE=harness-runs/improvement-1/harness.json \
  CONCURRENCY=1 USE_SHARED_DOCKER_BRIDGE=1 \
  bun run harbor:test10 -- t10s1-improvement-1 deepseek/deepseek-v4-flash

HARNESS_FILE=harness-runs/improvement-2/harness.json \
  CONCURRENCY=1 USE_SHARED_DOCKER_BRIDGE=1 \
  bun run harbor:test10 -- t10s1-improvement-2 deepseek/deepseek-v4-flash
```

新しい改善候補を作る場合は、`PROMPT.md` をコーディングエージェントに渡します。
このプロンプトは、`dev5` の trace だけを読み、`harness-runs/<name>/harness.json` だけを編集するように制約しています。

```bash
HARNESS_FILE=harness-runs/baseline/harness.json \
  CONCURRENCY=1 USE_SHARED_DOCKER_BRIDGE=1 \
  bun run harbor:dev5 -- dev5-baseline deepseek/deepseek-v4-flash
```

## Harborへ渡す情報

Harbor 系のコマンドは、Harbor に次の情報を渡します。

```text
harbor run
  -d terminal-bench/terminal-bench-2
  --agent-import-path adapters.harbor_agent:DeepAgentsTsHarborAgent
  --agent-kwarg harness_file=<HARNESS_FILE>
  --agent-kwarg trace_mode=<full|none>
  --agent-env BASE_MODEL=<model>
  -i <task-id> ...
  -k <N_ATTEMPTS>
  -n <CONCURRENCY>
  -o results/harbor
  --job-name tb2-<label>-<split>-<timestamp>
  --yes
  --debug
```

`USE_SHARED_DOCKER_BRIDGE=1` の場合だけ、次も渡します。

```text
--extra-docker-compose docker/docker-compose.shared-bridge.yaml
```

Harbor agent には次の環境変数も渡します。

- `OPENROUTER_API_KEY`
- `BASE_MODEL`
- `AGENT_WORKDIR`（指定されている場合）
- `AGENT_RUN_TIMEOUT_SEC`（指定されている場合）
- `OPENROUTER_PROVIDER_CONFIG`（指定されている場合）
- `DEEPAGENTS_TBENCH_REPO`（指定されている場合）
- `DEEPAGENTS_TBENCH_REF`（指定されている場合）

タスクコンテナ内では、agent adapter が次を行います。

- 35/ の実行に必要なファイルと `harness.json` を `/opt/deepagents-tbench-autotune-ts` に配置
- Harbor から受け取った task instruction を `/tmp/task.md` に保存
- タスクの作業ディレクトリを検出して `AGENT_WORKDIR` に設定
- `bun src/main.ts run /tmp/task.md /logs/agent/deepagents-ts harness.json` を実行
- 実行後に `bun src/main.ts report /logs/agent/deepagents-ts` を実行

`trace_mode=none` の場合、DeepAgents 側の `tool_events.jsonl`、`checkpoints.json`、`analysis_input.md` は生成しません。
Harbor の採点結果から `result.json` と `suite-summary.json` だけを残します。

## 対象タスクと評価タスク

`harbor:dev5` はチューニング用タスクです。
ハーネス調整時に trace を読んでよいのは、この5タスクだけです。

```text
terminal-bench/prove-plus-comm
terminal-bench/fix-git
terminal-bench/openssl-selfsigned-cert
terminal-bench/log-summary-date-ranges
terminal-bench/regex-log
```

`harbor:test10` は評価用タスクです。
Terminal-Bench 2 に存在し、チューニング用タスクと重複せず、短時間で完了しやすいタスクを優先して選定しています。

```text
terminal-bench/cobol-modernization      # easy, COBOLロジックのPython再実装
terminal-bench/crack-7z-hash            # medium, 7z復号
terminal-bench/raman-fitting            # medium, 数値フィット
terminal-bench/kv-store-grpc            # medium, gRPCサービス実装
terminal-bench/merge-diff-arc-agi-task  # medium, Git mergeとデータ変換
terminal-bench/extract-elf              # medium, ELF解析とファイル処理
terminal-bench/sqlite-with-gcov         # medium, SQLiteのgcovビルド
terminal-bench/pypi-server              # medium, Python packageとローカルPyPI
terminal-bench/sqlite-db-truncate       # medium, SQLite破損DB復旧
terminal-bench/polyglot-c-py            # medium, C/Python polyglot実装
```

## 結果の場所

Harbor の実行結果は `results/harbor/<job>/` に保存されます。

`harbor:test10` の結果は、比較に必要なファイルだけを残します。

- `result.json`: Harbor の採点結果
- `suite-summary.json`: trial平均と task別結果の集計
- `test10-comparison.md`: `baseline -> improvement-1 -> improvement-2` の比較表

`harbor:dev5` のように `trace_mode=full` で実行した場合は、taskごとの agent log も保存されます。
trace を読むのは、ハーネス改善案を作るためのチューニング用実行に限ります。

直近の `test10` job id は次に追記されます。

```text
results/harbor/latest-tb2-test10.txt
```

`suite-summary.json` では、各 trial の平均に加えて、同じタスクの全 trial が成功した場合だけ成功と数える `strictTaskMean` を確認します。
ただし、`harbor:test10` は1回試行なので、`strictTaskPasses` は `trialPasses` と同じ値になります。
`harbor:dev5` の標準設定では 3 trial、`harbor:test10` では 1 trial 固定です。

## ローカルスモークテスト

公式 dataset を使う前に、ローカルの小さな Terminal-Bench タスクで接続を確認できます。

```bash
bun run tbench:smoke
```

ローカルスモークテスト用の task は `smoke_tasks/` にあります。

## Runner単体で確認する

Harbor を使わずに runner だけを実行する場合は、task file、出力先、harness file を指定します。

```bash
mkdir -p tmp/manual-workspace results/manual

cat > tmp/manual-task.md <<'EOF'
Create a file named answer.txt containing exactly:
terminal-bench harness
Then verify the content using a shell command.
EOF

AGENT_WORKDIR="$PWD/tmp/manual-workspace" \
  bun run run -- tmp/manual-task.md results/manual/round0 harness-runs/baseline/harness.json

bun run report -- results/manual/round0
```

## ファイル構成

```text
35/
├── adapters/                 # Harbor / Terminal-Bench agent adapter
├── docker/                   # optional Docker Compose overlay
├── harness-runs/             # harness.json variants
├── results/                  # run outputs
├── scripts/                  # execution scripts
├── src/                      # Bun / TypeScript runner
├── smoke_tasks/              # local smoke test tasks
├── .env.sample
├── .mise.toml
├── package.json
└── tsconfig.json
```

## 確認コマンド

```bash
bun run check
bash -n scripts/run_harbor_tb2_dev5.sh scripts/run_harbor_tb2_test10.sh scripts/run_tbench_smoke.sh scripts/container-setup.sh
```

## 参考リンク

- [Harbor Terminal-Bench tutorial](https://www.harborframework.com/docs/tutorials/running-terminal-bench)
- [LangChain JS DeepAgents docs](https://docs.langchain.com/oss/javascript/deepagents/overview)
- [LangChain ChatOpenRouter integration](https://docs.langchain.com/oss/javascript/integrations/chat/openrouter)
- [OpenRouter DeepSeek V4 Flash](https://openrouter.ai/deepseek/deepseek-v4-flash)
