# Software Design誌「実践LLMアプリケーション開発」第36回サンプルコード

第32回の「arXiv論文→スライド生成」ワークフローを、DeepEvalの搭載済みメトリクスだけで評価するサンプルです。
エージェント本体はTypeScript(deepagents)、評価は薄いPython層(DeepEval)に分離しています。
スキル指示だけが異なる2つのバリアント(`baseline` / `improved`)を同じメトリクスで測り、「計測→改善→再計測」のループを再現できます。

## 前提条件

- [mise](https://mise.jdx.dev/)
- Bun
- `uv`
- OpenRouter APIキー(エージェント実行: `deepseek/deepseek-v4-flash`)
- OpenAI APIキー(評価judge: `gpt-5.4`)

## セットアップ

```bash
mise trust 36/.mise.toml
cd 36
mise install
mise run install
```

`.env` を作成し、APIキーを設定します。

```bash
cp .env.sample .env
vi .env
```

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

- `OPENROUTER_API_KEY`: エージェント実行用(OpenRouter経由でdeepseek-v4-flashを呼び出す)
- `OPENAI_API_KEY`: DeepEvalのjudgeモデル用(gpt-5.4)

## エージェントの実行(ヘッドレスランナー)

第32回は対話アプリでしたが、評価のためにヘッドレスランナーで同じワークフローを再現します。
ランナーは2ターンで実行します。人間によるアウトライン確認を、固定の承認メッセージで置き換えます。

1. ターン1: 論文URLを渡す → エージェントが論文を取得・分析し、アウトラインを提案する
2. ターン2: 「OKです。この構成でスライドを生成してください。」→ `generate_pptx` ツールで生成する

```bash
bun run agent 1706.03762 baseline
bun run agent 1706.03762 improved
```

実行結果は `results/<variant>/<arXiv ID>.json` に保存されます。
スライドJSON・実行中のツール呼び出し・所要時間が入っており、評価はこのファイルだけを読みます。

記事の実験で使ったデータセットは次の3本です。

```bash
for id in 1706.03762 2512.07828 2603.03303; do
  bun run agent "$id" baseline
  bun run agent "$id" improved
done
```

## 評価の実行

```bash
uv run eval/run_eval.py baseline
uv run eval/run_eval.py improved
```

judgeのブレを確認する場合は、対象を絞って繰り返し実行します。

```bash
uv run eval/run_eval.py baseline --id 1706.03762 --repeat 3
```

スコアと理由は `results/eval/<variant>.json` に保存されます。

## 評価メトリクス

すべてDeepEvalに搭載済みのメトリクスです。カスタムメトリクスは実装していません。

| 段階 | メトリクス | 入力 | 測るもの |
|---|---|---|---|
| 取得 | Tool Correctness | tools_called vs expected_tools | 期待したワークフロー(論文取得→generate_pptx)を通ったか |
| 要約中核 | Summarization | input=論文本文, actual_output=スライドテキスト | 整合性×網羅性(score = min(両者)) |
| 見せ方 | G-Eval(SlideQuality) | input, actual_output | 1スライド1論点・論理的な流れ・タイトルの情報量 |

- 理想スライド(gold)は用意しません。Summarizationは「原文→要約」を直接測るreference-freeなメトリクスです
- 論文本文は評価側が [ar5iv](https://ar5iv.labs.arxiv.org/) から取得します(2023年以前の論文はarxiv.org/htmlが未対応のため)
- judgeは全メトリクスで `gpt-5.4` に統一しています。G-Evalはスコアをトークンのlogprobsで加重するため、logprobs対応モデルが必要です

## スキルバリアント

`workspaces/baseline` と `workspaces/improved` は、スキル(SKILL.md)の指示だけが異なります。

```bash
diff -r workspaces/baseline workspaces/improved
```

- `baseline`: 第32回のスキルに、古い論文向けのar5ivフォールバックだけを追加したもの
- `improved`: ベースライン計測で見えた弱点(要約の忠実性・詰め込み)に対する改善を追加したもの
  - 保存前の事実確認(数値・構成要素の数を本文の記述と照合する)
  - 実験結果の一般化禁止(対象タスク・データセット・条件を本文の通りに限定する)
  - 照合できなかった数値・固有名詞は書かない(定性的表現に置き換えるか削除)
  - 1スライド1メッセージ(箇条書きは最大4項目・各1行)
  - スライドタイトルに主張を含める
  - 主要な数値結果・計算資源を漏らさない

どちらのバリアントも実行モデルは `deepseek/deepseek-v4-flash` で共通です。測定される差分はスキル指示の質だけです。

## ファイル構成

```text
36/
├── agent/                      # 第32回から流用したエージェント本体
│   ├── agent.ts                # createDeepAgent定義(モデルはOpenRouter経由に変更)
│   ├── generate-pptx-tool.ts   # generate_pptxツール(スキーマ検証内蔵)
│   └── system-prompt.ts        # システムプロンプト
├── agent-run/
│   └── run.ts                  # ヘッドレスランナー(2ターン実行・成果物の書き出し)
├── workspaces/
│   ├── baseline/               # 第32回相当のスキル
│   └── improved/               # 改善版スキル
├── eval/
│   ├── cases.py                # results/*.json + ar5iv本文 → LLMTestCase
│   ├── metrics.py              # 搭載済みメトリクスの組み立て
│   └── run_eval.py             # evaluate() 実行・スコア保存
├── results/
│   ├── baseline/               # ランナー成果物
│   ├── improved/
│   └── eval/                   # 評価スコアと理由
├── package.json
├── pyproject.toml
├── .mise.toml
└── .env.sample
```

## 実験結果

2026-07-08時点の実測値です。各バリアントで3論文を実行し、judgeを3回繰り返した平均を示します(カッコ内は3回のブレ幅)。

| メトリクス | baseline | improved |
|---|---|---|
| Tool Correctness | 1.000 | 1.000 |
| Summarization | 0.558 | **0.639** |
| SlideQuality (G-Eval) | 0.801 | 0.777 |

論文別のSummarization平均:

| 論文 | baseline | improved |
|---|---|---|
| 1706.03762 (Attention) | 0.602 (0.55-0.67) | **0.783 (0.75-0.80)** |
| 2512.07828 | 0.451 (0.33-0.52) | **0.549 (0.52-0.60)** |
| 2603.03303 | 0.622 (0.45-0.75) | 0.584 (0.50-0.68) |

- ベースラインの弱点はSummarization(要約の忠実性×網羅性)で、手順(Tool Correctness)と見せ方(G-Eval)は最初から良好でした
- improvedでは狙った弱点だけが改善し、Attentionでは判定のブレも縮小しています(事実確認がスコアを上げ、かつ安定させた)
- Summarizationの判定は実行ごとに±0.1程度ブレます。単発のスコアではなく複数回の平均で比較してください
- 評価側の失敗談: 当初はsourceから`<math>`要素を丸ごと除去していたため、本文由来の数値・数式がスライド側で「原文にない付け足し」と誤判定されました。現在はalttext属性(LaTeX)へ置き換えて残しています(`eval/cases.py`)。reference-free評価ではsourceの忠実性がそのまま判定の質を決めます

## 確認コマンド

```bash
bun run check
```

## 参考リンク

- [DeepEval documentation](https://deepeval.com/docs/getting-started)
- [LangChain JS DeepAgents docs](https://docs.langchain.com/oss/javascript/deepagents/overview)
- [OpenRouter DeepSeek V4 Flash](https://openrouter.ai/deepseek/deepseek-v4-flash)
- [ar5iv](https://ar5iv.labs.arxiv.org/)
