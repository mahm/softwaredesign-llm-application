# 第40回 サンプルコード実行ガイド

本ディレクトリには、Software Design連載「実践LLMアプリケーション開発」第40回で解説しているサンプルコードを収録しています。
TypeSafeの意思決定モデル（decision model）であるJevをRAG（検索拡張生成）パイプラインに組み込み、検索対象のカテゴリ絞り込み、候補チャンクの一括リランキング、スコア閾値による選別と前後チャンクの再構成を行う方法を確認できます。

## 動作環境

- ランタイム: Node.js 26.13.1
- 言語: TypeScript 7.0.2
- 主要ライブラリ: `@typesafe-ai/sdk` 0.7.0, `@openrouter/sdk` 1.5.100
- 使用モデル:
  - 意思決定モデル: `jev-1.14.0`（TypeSafe）
  - 埋め込みモデル: `openai/text-embedding-3-small`（OpenRouter経由）
  - テキスト生成モデル: `deepseek/deepseek-v4.1-flash-1017`（OpenRouter経由）

Node.jsのTypeScript実行機能で、各サンプルを直接実行します。

## 準備

依存パッケージをインストールします。

```bash
cd 40
npm ci
```

型を検査して問題がないことを確認します。

```bash
npm run check
```

環境変数`TYPESAFE_API_KEY`にTypeSafeのAPIキー、`OPENROUTER_API_KEY`にOpenRouterのAPIキーを設定します。
サンプルの実行には、TypeSafeおよびOpenRouterの利用料金がかかります。

## データ

`inquiries/sdk-example.txt`は、第39回と同じ架空の社内問い合わせです。
新しいノートPCで自宅からVPNに接続しようとした際、接続画面から先に進めないという症状と、初回設定・接続方法を知りたいという質問が書かれています。

`docs/chunks.json`は、社内ヘルプデスクで参照する4つの文書カテゴリ（`it_manual`：IT機器・ネットワーク利用手順書、`security_policy`：情報セキュリティ規定、`hr_general`：総務・労務ガイド、`incident_log`：過去の障害・メンテナンス履歴）のチャンクデータ（全10件）です。

## 掲載サンプルコード一覧と実行方法

以下のコマンドは`40/`ディレクトリで実行します。ファイル名の番号は、記事で紹介する順序です。

| ファイル名 | 概要 | 実行コマンド例 |
|---|---|---|
| [`search.ts`](samples/search.ts) | OpenRouterのEmbeddings APIによるベクトル検索とBM25全文検索をRRF（Reciprocal Rank Fusion）で統合するハイブリッド検索ヘルパー | （各サンプルから呼び出し） |
| [`01-filter-collection.ts`](samples/01-filter-collection.ts) | `noul`で問い合わせから検索すべき文書カテゴリを判定し、絞り込み前後のハイブリッド検索結果を比較 | `node samples/01-filter-collection.ts` |
| [`02-rerank-chunks.ts`](samples/02-rerank-chunks.ts) | 候補チャンクを1回のリクエストで評価（`score`：無関係／手がかり／答え）し、期待値スコアでリランキング | `node samples/02-rerank-chunks.ts` |
| [`03-reconstruct-and-answer.ts`](samples/03-reconstruct-and-answer.ts) | スコア閾値でチャンクを選別し、同一セクションの前後チャンクを再構成してOpenRouter経由のLLMで回答を生成 | `node samples/03-reconstruct-and-answer.ts` |

同じ入力に対する回答と確率の変動によって、実行結果が記事の掲載内容と異なる場合があります。
