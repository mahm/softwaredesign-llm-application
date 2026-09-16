# 第38回 サンプルコード実行ガイド

本ディレクトリには、Software Design連載「実践LLMアプリケーション開発」第38回で解説しているサンプルコードを収録しています。
OpenRouterのレスポンスキャッシュ、Server Tools（Web検索）、PDF Inputs、レスポンスヒーリング、コンテキスト圧縮の動作を手元で確認できます。

## 動作環境

- ランタイム: Bun 1.3.5
- 言語: TypeScript 6.0.3
- 主要ライブラリ: `@openrouter/sdk` 1.2.116, `zod` 4.4.3
- 使用モデル: `deepseek/deepseek-v4-flash-0731`

## 準備

依存パッケージをインストールします。

```bash
cd 38
bun install
```

型を検査して問題がないことを確認します。

```bash
bun run check
```

実行プロセスの環境変数`OPENROUTER_API_KEY`へOpenRouterのAPIキーを渡します。
Web検索や有料OCR（Mistral OCR）の実行には料金がかかるため、アカウントのクレジット残高を確認してください。

## 掲載サンプルコード一覧と実行方法

以下のコマンドは`samples/`ディレクトリで実行します。

```bash
cd samples
```

記事で解説している主要なコードは以下の6本です。Server ToolsのWeb検索は2026年9月時点でベータ提供の機能です。

| ファイル名 | 概要 | 実行コマンド例 |
|---|---|---|---|
| [`company-news-cache.ts`](samples/company-news-cache.ts) | 企業ニュース照会とレスポンスキャッシュ（TTL 900秒、Exa Web検索、2回連続呼び出し） | `bun company-news-cache.ts ソニーグループ` |
| [`response-cache.ts`](samples/response-cache.ts) | アカウントZDR設定によるキャッシュ動作の比較（短文呼び出し、2回連続送信） | `bun response-cache.ts` |
| [`pdf-input.ts`](samples/pdf-input.ts) | テキストPDFまたはスキャンPDFからの5項目抽出（デフォルトはCloudflare AI、スキャンはMistral OCR） | `bun pdf-input.ts cloudflare-ai ../fixtures/pdf/quote-text.pdf`<br>`bun pdf-input.ts mistral-ocr ../fixtures/pdf/quote-scan.pdf` |
| [`extras-pdf-reuse.ts`](samples/extras-pdf-reuse.ts) | 抽出済みannotationsを引き継いだPDF追加質問 | `bun extras-pdf-reuse.ts`<br>`bun extras-pdf-reuse.ts --compare` |
| [`response-healing.ts`](samples/response-healing.ts) | JSONレスポンスの構文修復（通常実行、およびstopによる括弧欠落の修復比較） | `bun response-healing.ts --normal`<br>`bun response-healing.ts` |
| [`context-compression.ts`](samples/context-compression.ts) | 会話履歴のmiddle-out圧縮（短い会話、圧縮無効での超過エラー、圧縮有効で中間の値が抜ける動作の確認） | `bun context-compression.ts short`<br>`bun context-compression.ts off`<br>`bun context-compression.ts on` |
