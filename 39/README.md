# 第39回 サンプルコード実行ガイド

本ディレクトリには、Software Design連載「実践LLMアプリケーション開発」第39回で解説しているサンプルコードを収録しています。
TypeSafeの意思決定モデル（decision model）であるJevを使い、社内ヘルプデスクに届いた問い合わせを分類します。

## 動作環境

- ランタイム: Bun 1.4.2
- 言語: TypeScript 7.0.2
- 主要ライブラリ: `@typesafe-ai/sdk` 0.6.0
- 使用モデル: `jev-latest`（2026年10月時点では`jev-1.13.0`）

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

環境変数`TYPESAFE_API_KEY`にTypeSafeのAPIキーを設定します。
サンプルの実行には、TypeSafeの利用料金がかかります。

## データ

| ファイル名 | 内容 |
|---|---|
| `inquiries/001.txt`〜`030.txt` | 社内ヘルプデスクに届いた社員からの問い合わせを想定して作成した、架空の問い合わせ30件 |
| `inquiries/received-at.tsv` | 各問い合わせの受付日時 |
| `labels.tsv` | 著者が付けた正解ラベル。`05-evaluate.ts`で、Jevの回答と比べるために使います |

`labels.tsv`の列名は、`03-inquiry-questions.ts`の質問のキーと同じです。
複数の担当部署にまたがる問い合わせ（010、028）は、用件ごとに1行ずつ正解を記載しています。
`notes`の列には、正解を付けるときに別の解釈も成り立った点を書いています。

## 掲載サンプルコード一覧と実行方法

以下のコマンドは`39/`ディレクトリで実行します。ファイル名の番号は、記事で紹介する順序です。`05-evaluate.ts`は、記事の実行結果の集計に使ったスクリプトです。

| ファイル名 | 概要 | 実行コマンド例 |
|---|---|---|
| [`01-basic-choice.ts`](samples/01-basic-choice.ts) | `choice`の質問1つで問い合わせ001の種別を判断させ、送るJSONと受け取るJSONを表示 | `bun samples/01-basic-choice.ts` |
| [`02-basic-questions.ts`](samples/02-basic-questions.ts) | `choice`、`score`、`noul`の3種類の質問を1回のリクエストで送り、送るJSONと受け取るJSONを表示 | `bun samples/02-basic-questions.ts` |
| [`03-inquiry-questions.ts`](samples/03-inquiry-questions.ts) | 分類に使う7つの質問の定義。`04-classify.ts`と`05-evaluate.ts`から読み込みます | — |
| [`04-classify.ts`](samples/04-classify.ts) | 30件を分類し、種別、対象、担当部署、優先度、複数部署、要確認をタブ区切りで表示 | `bun samples/04-classify.ts` |
| [`05-evaluate.ts`](samples/05-evaluate.ts) | 30件を指定した回数ずつ分類して正解ラベルと比べ、記事の実行結果（表5〜表11）に当たる値を表示。質問ごとの一致、担当部署の振り分けの結果、優先度と要確認が正解ラベルと異なった問い合わせ、要確認のしきい値ごとの誤り、所要時間と入力トークン数 | `bun samples/05-evaluate.ts`<br>`bun samples/05-evaluate.ts 1` |

`05-evaluate.ts`は、30件×指定した回数のリクエストを送ります。回数を省略した場合は10回ずつ、合わせて300リクエストを送ります。
`jev-latest`が指すモデルの更新や、同じ入力に対する回答と確率の変動によって、実行結果が記事の掲載内容と異なる場合があります。
