# 第39回 サンプルコード実行ガイド

本ディレクトリには、Software Design連載「実践LLMアプリケーション開発」第39回で解説しているサンプルコードを収録しています。
TypeSafeの意思決定モデル（decision model）であるJevの質問タイプと、SDKの基本的な使い方を紹介します。

## 動作環境

- ランタイム: Node.js 24.22.0（LTS）
- 言語: TypeScript 7.0.2
- 主要ライブラリ: `@typesafe-ai/sdk` 0.7.0
- 使用モデル: `jev-1.14.0`

Node.jsのTypeScript実行機能で、各サンプルを直接実行します。

## 準備

依存パッケージをインストールします。

```bash
cd 39
npm ci
```

型を検査して問題がないことを確認します。

```bash
npm run check
```

環境変数`TYPESAFE_API_KEY`にTypeSafeのAPIキーを設定します。
サンプルの実行には、TypeSafeの利用料金がかかります。

## データ

`inquiries/sdk-example.txt`は、質問タイプの説明で使用する架空の問い合わせです。
新しいPCからVPNに接続できないという症状と、接続方法を知りたいという質問が書かれています。

`assets/typesafe-homepage.png`は、2026年10月15日に撮影した[TypeSafe公式サイト](https://typesafe.ai/)のスクリーンショットです。
記事の図2と同じ画像を使い、「Create Account」リンクの有無と位置を調べます。

## 掲載サンプルコード一覧と実行方法

以下のコマンドは`39/`ディレクトリで実行します。ファイル名の番号は、記事で紹介する順序です。

| ファイル名 | 概要 | 実行コマンド例 |
|---|---|---|
| [`01-basic-choice.ts`](samples/01-basic-choice.ts) | `choice`で問い合わせの種別を判断させ、送るJSONと受け取るJSONを表示 | `node samples/01-basic-choice.ts` |
| [`02-basic-questions.ts`](samples/02-basic-questions.ts) | 緊急度を`score`、対応前の確認の要否を`noul`で尋ね、送るJSONと受け取るJSONを表示 | `node samples/02-basic-questions.ts` |
| [`03-image-box.ts`](samples/03-image-box.ts) | 画像内のリンクの有無を`noul`、位置を`box`で尋ね、受け取るJSONを表示 | `node samples/03-image-box.ts` |

画像のサンプルでは、`TypeSafe-Danger-Zone-Images: true`ヘッダーを指定しています。
`box`は対象が画像内に存在しない場合でも座標を返すため、`create_account_visible`の`noul`も確認してください。

同じ入力に対する回答と確率の変動によって、実行結果が記事の掲載内容と異なる場合があります。
