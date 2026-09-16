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

| ファイル名 | 要求数 | 概要 | 実行コマンド例 |
|---|---|---|---|
| [`company-news-cache.ts`](samples/company-news-cache.ts) | 2 | 企業ニュース照会とレスポンスキャッシュ（TTL 900秒、Exa Web検索、2回連続呼び出し） | `bun company-news-cache.ts ソニーグループ` |
| [`response-cache.ts`](samples/response-cache.ts) | 2 | アカウントZDR設定によるキャッシュ動作の比較（短文呼び出し、2回連続送信） | `bun response-cache.ts` |
| [`pdf-input.ts`](samples/pdf-input.ts) | 1 | テキストPDFまたはスキャンPDFからの5項目抽出（デフォルトはCloudflare AI、スキャンはMistral OCR） | `bun pdf-input.ts`<br>`bun pdf-input.ts mistral-ocr ../fixtures/pdf/quote-scan.pdf` |
| [`extras-pdf-reuse.ts`](samples/extras-pdf-reuse.ts) | 2（`--compare`は3） | 抽出済みannotationsを引き継いだPDF追加質問 | `bun extras-pdf-reuse.ts`<br>`bun extras-pdf-reuse.ts --compare` |
| [`response-healing.ts`](samples/response-healing.ts) | 2（`--normal`は1） | JSONレスポンスの構文修復（通常実行、およびstopによる括弧欠落の修復比較） | `bun response-healing.ts --normal`<br>`bun response-healing.ts` |
| [`context-compression.ts`](samples/context-compression.ts) | 各1 | 会話履歴のmiddle-out圧縮（短い会話、圧縮無効での超過エラー、圧縮有効で中間の値が抜ける動作の確認） | `bun context-compression.ts short`<br>`bun context-compression.ts off`<br>`bun context-compression.ts on` |

### 実行時の動作と確認点

#### キャッシュの一致条件

同じ企業名、同じ日付（JST）、同じAPIキーでリクエスト本文が一致し、キャッシュが保存されている間に再利用されます。
同時リクエストを1つにまとめる機能はありません。
TTLに指定した900秒は保存期間の上限であり、サーバーの状況によって期限前に消去される場合もあります。
期限が切れたあとのリクエストでは新しく回答が生成されます。
なお、検索ツールを実行するかどうかはモデルが判断します。
今回のコードではExa検索の最大回数を`maxUses: 1`に設定しています。

#### ZDRの影響

アカウント設定（Privacy）でZDRが有効な場合、モデルは通常どおり回答しますが、レスポンスヘッダーにキャッシュ状態（`x-openrouter-cache-status`）が含まれません（`null`）。
ZDRが無効に設定されており、リクエストの内容が一致していれば、有効な回答が保存されているあいだは`HIT`となります。
このときは消費トークン数と料金が0になります。

#### 実行時間と使用量情報

`companyNews`では、呼び出し前後の`performance.now()`の差分から処理にかかった経過秒数（`elapsedSeconds`）を計測して返します。
レスポンスヘッダーからは`cacheStatus`と保存後の経過秒数（`cacheAge`）を取得します。
HIT時にも`created`フィールドは更新されるため、検索時刻の判定には使えません。
また、SDKオブジェクトの`usage.serverToolUseDetails.webSearchRequests`には保存元の値が残るため、再検索が行われた証拠と誤認しないよう注意してください。

#### PDF解析結果の引き継ぎ

使用しているSDK 1.2.116ではPDF解析結果の`annotations`が送受信時に保持されないため、[`extras-pdf-reuse.ts`](samples/extras-pdf-reuse.ts)では`fetch`による直接呼び出しを使用しています。
PDFファイル（`../fixtures/pdf/`）の相対配置を変更せずに実行してください。

#### 固定値の照合

[`pdf-input.ts`](samples/pdf-input.ts)ではZodでキーと型を検証したあと、付属フィクスチャの固定値と`Object.entries`で比較しています。
[`response-healing.ts`](samples/response-healing.ts)ではZodの`z.literal`を使って固定値まで照合しています。
別の入力データを与える場合は、これらの照合ロジックを変更してください。

#### コンテキスト圧縮の挙動

`context-compression.ts`では、圧縮無効（`off`）でコンテキスト長を超過した際にHTTP 400エラーが発生します。
今回のサンプルで圧縮を有効（`on`）にした実行では、エラーを回避できた一方で、中央に配置した数量情報が`null`として返る結果を確認しました。
