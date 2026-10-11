# 第39回 サンプルコード実行ガイド

本ディレクトリには、Software Design連載「実践LLMアプリケーション開発」第39回で解説しているサンプルコードを収録しています。
TypeSafeの意思決定モデル（decision model）であるJevを使い、社内ヘルプデスクに届いた問い合わせを分類します。
TypeSafeのSDKで呼び出す例のほか、OpenRouterのDecisions APIからJevとGPT-6 Luna Decisions（`openai/gpt-6-luna-decisions`）を呼び出す例も含みます。

## 動作環境

- ランタイム: Bun 1.4.2
- 言語: TypeScript 7.0.2
- 主要ライブラリ: `@typesafe-ai/sdk` 0.6.0, `@openrouter/sdk` 1.4.25
- 使用モデル: `jev-latest`（2026年10月時点では`jev-1.13.0`）、`typesafe/jev-1.13`、`openai/gpt-6-luna-decisions`

TypeSafeのSDKの資料が示す実行環境はNode.js 20以上です。本サンプルはBun 1.4.2で動作を確認しています。

## 準備

依存パッケージをインストールします。

```bash
cd 39
bun install
```

型を検査して問題がないことを確認します。

```bash
bun run check
```

環境変数`TYPESAFE_API_KEY`にTypeSafeのAPIキーを、`OPENROUTER_API_KEY`にOpenRouterのAPIキーを設定します。
サンプルの実行には、TypeSafeとOpenRouterの利用料金がかかります。

## データ

| ファイル名 | 内容 |
|---|---|
| `inquiries/001.txt`〜`030.txt` | 社内ヘルプデスクに届いた社員からの問い合わせを想定して作成した、架空の問い合わせ30件 |
| `inquiries/received-at.tsv` | 各問い合わせの受付日時 |
| `labels.tsv` | 著者が付けた正解ラベル。`06-compare-models.ts`で、モデルの回答と比べるために使います |

`labels.tsv`の列名は、`03-inquiry-questions.ts`の質問のキーと同じです。
複数の担当部署にまたがる問い合わせ（010、028）は、用件ごとに1行ずつ正解を記載しています。
`notes`の列には、正解を付けるときに別の解釈も成り立った点を書いています。

## 掲載サンプルコード一覧と実行方法

以下のコマンドは`39/`ディレクトリで実行します。ファイル名の番号は、記事で紹介する順序です。

| ファイル名 | 概要 | 実行コマンド例 |
|---|---|---|
| [`01-basic-choice.ts`](samples/01-basic-choice.ts) | `choice`の質問1つで問い合わせ001の種別を判断させ、レスポンス全体をJSONで表示 | `bun samples/01-basic-choice.ts` |
| [`02-basic-questions.ts`](samples/02-basic-questions.ts) | `choice`、`score`、`noul`の3種類の質問を1回のリクエストで送り、回答と確率、確信度を表示 | `bun samples/02-basic-questions.ts` |
| [`03-inquiry-questions.ts`](samples/03-inquiry-questions.ts) | 分類に使う7つの質問の定義。`04-classify.ts`と`06-compare-models.ts`から読み込みます | — |
| [`04-classify.ts`](samples/04-classify.ts) | 30件を分類し、種別、対象、担当部署、優先度、複数部署、要確認をタブ区切りで表示 | `bun samples/04-classify.ts` |
| [`05-openrouter-basic.ts`](samples/05-openrouter-basic.ts) | OpenRouterのDecisions APIで種別の質問を送る。モデルIDは引数で指定（省略時は`typesafe/jev-1.13`） | `bun samples/05-openrouter-basic.ts`<br>`bun samples/05-openrouter-basic.ts openai/gpt-6-luna-decisions` |
| [`06-compare-models.ts`](samples/06-compare-models.ts) | OpenRouterのDecisions APIで、指定したモデルに30件を指定した回数ずつ分類させ、質問ごとの正解ラベルとの一致数、回答が一度でも変わった問い合わせの数、所要時間の中央値、利用料金の合計を表示 | `bun samples/06-compare-models.ts typesafe/jev-1.13 10`<br>`bun samples/06-compare-models.ts openai/gpt-6-luna-decisions 10` |

`06-compare-models.ts`は、30件×指定した回数のリクエストを送ります。回数を省略した場合は1回ずつ送ります。
`jev-latest`が指すモデルの更新や、同じ入力に対する回答の揺れによって、実行結果が記事の掲載内容と異なる場合があります。
