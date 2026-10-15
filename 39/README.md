# 第39回 サンプルコード実行ガイド

本ディレクトリには、Software Design連載「実践LLMアプリケーション開発」第39回で解説しているサンプルコードを収録しています。
TypeSafeの意思決定モデル（decision model）であるJevの質問タイプと、SDKの基本的な使い方を紹介します。

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

`inquiries/001.txt`は、質問タイプの説明で使用する架空の問い合わせです。

## 掲載サンプルコード一覧と実行方法

以下のコマンドは`39/`ディレクトリで実行します。ファイル名の番号は、記事で紹介する順序です。

| ファイル名 | 概要 | 実行コマンド例 |
|---|---|---|
| [`01-basic-choice.ts`](samples/01-basic-choice.ts) | `choice`の質問1つで問い合わせ001の種別を判断させ、送るJSONと受け取るJSONを表示 | `bun samples/01-basic-choice.ts` |
| [`02-basic-questions.ts`](samples/02-basic-questions.ts) | `choice`、`score`、`noul`の3種類の質問を1回のリクエストで送り、送るJSONと受け取るJSONを表示 | `bun samples/02-basic-questions.ts` |

`jev-latest`が指すモデルの更新や、同じ入力に対する回答と確率の変動によって、実行結果が記事の掲載内容と異なる場合があります。
