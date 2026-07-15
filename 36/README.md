# Software Design誌「実践LLMアプリケーション開発」第36回サンプルコード

第32回の「arXiv論文→スライド生成」ワークフローを題材に、DeepEvalの搭載済みメトリクスだけで「計測→改善→再計測」のループを回すサンプルです。
エージェント本体はTypeScript(deepagents)、評価はPython(DeepEval)で実装しています。

3つのスキルの作り込み段階を持ち、各改善を直前の段階における改善と比較します。

```
baseline        スキルはワークフローの機構のみ(取得手順・JSON形式・枚数・確認フロー)
  ↓ +スライド設計ガイド                → G-Eval(見せ方)で比較
improvement-1   詰め込み禁止・主張型タイトル・論理的な流れ
  ↓ +保存前の事実確認                  → Summarization(忠実性×網羅)で比較
improvement-2   本文照合・一般化禁止・照合できない数値は書かない
```

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

1. ターン1: 論文URLを渡す → エージェントが論文を取得・分析し、アウトラインを提案する
2. ターン2: 「OKです。この構成でスライドを生成してください。」→ `generate_pptx` ツールで生成する

```bash
bun run agent 1706.03762 baseline
bun run agent 1706.03762 improvement-1
bun run agent 1706.03762 improvement-2
```

実行結果は `results/<variant>/<arXiv ID>.json` に保存されます。
スライドJSON・実行中のツール呼び出し(サブエージェント内を含む)・所要時間が入っており、評価はこのファイルだけを読みます。

記事の実験で使ったデータセットは次の3本です。

```bash
for id in 1706.03762 2512.07828 2603.03303; do
  bun run agent "$id" baseline
  bun run agent "$id" improvement-1
  bun run agent "$id" improvement-2
done
```

### 使用論文

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. "Attention Is All You Need." NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Jeremy Yang, Noah Yonack, Kate Zyskowski, Denis Yarats, Johnny Ho, Jerry Ma. "The Adoption and Usage of AI Agents: Early Evidence from Perplexity." 2025. [arXiv:2512.07828](https://arxiv.org/abs/2512.07828)
- Shirley Wu, Evelyn Choi, Arpandeep Khatua, Zhanghan Wang, Joy He-Yueya, Tharindu Cyril Weerasooriya, Wei Wei, Diyi Yang, Jure Leskovec, James Zou. "HumanLM: Simulating Users with State Alignment Beats Response Imitation." 2026. [arXiv:2603.03303](https://arxiv.org/abs/2603.03303)

## 評価の実行

```bash
uv run eval/run_eval.py baseline --repeat 3
uv run eval/run_eval.py improvement-1 --repeat 3
uv run eval/run_eval.py improvement-2 --repeat 3
```

スコアと理由は `results/eval/<variant>.json` に保存されます。
`--repeat` はjudgeのブレを見るための繰り返し実行です。単発のスコアではなく複数回の平均で比較してください。

## ファイル構成

```text
36/
├── agent/                      # 第32回から流用したエージェント本体
│   ├── agent.ts                # createDeepAgent定義(モデルはOpenRouter経由に変更)
│   ├── generate-pptx-tool.ts   # generate_pptxツール(スキーマ検証内蔵)
│   └── system-prompt.ts        # システムプロンプト
├── agent-run/
│   └── run.ts                  # ヘッドレスランナー
├── workspaces/
│   ├── baseline/               # 機構のみのスキル
│   ├── improvement-1/          # +スライド設計ガイド
│   └── improvement-2/          # +保存前の事実確認
├── eval/
│   ├── cases.py                # results/*.json + ar5iv本文 → LLMTestCase
│   ├── metrics.py              # 搭載済みメトリクスの組み立て
│   └── run_eval.py             # evaluate() 実行・スコア保存
├── results/
│   ├── baseline/               # ランナー成果物
│   ├── improvement-1/
│   ├── improvement-2/
│   └── eval/                   # 評価スコアと理由
├── package.json
├── pyproject.toml
├── .mise.toml
└── .env.sample
```

## 確認コマンド

```bash
bun run check
```

## 参考リンク

- [DeepEval documentation](https://deepeval.com/docs/getting-started)
- [LangChain JS DeepAgents docs](https://docs.langchain.com/oss/javascript/deepagents/overview)
- [OpenRouter DeepSeek V4 Flash](https://openrouter.ai/deepseek/deepseek-v4-flash)
- [ar5iv](https://ar5iv.labs.arxiv.org/)
