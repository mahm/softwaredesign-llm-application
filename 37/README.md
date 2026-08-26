# 第37回 OpenRouter入門 サンプルアプリケーション

## 動作環境

- Bun 1.3.5
- TypeScript 6.0.3
- @openrouter/sdk 1.2.51
- deepagents 1.12.4
- @langchain/openrouter 0.4.8
- Claude Code 2.1.239

## 準備

次のコマンドで依存パッケージをインストールします。

```console
bun install
```

Claude Codeは依存パッケージに含まれないため、[公式のインストール手順](https://code.claude.com/docs/en/installation)に従ってCLIを別途インストールします。
本サンプルの動作確認には2.1.239を使用しています。

```console
claude --version
```

`.env.example`を`.env`へコピーし、OpenRouterのAPIキーを設定します。

```console
cp .env.example .env
```

各コマンドはOpenRouterのAPIを呼び出すため、利用料金が発生します。

## OpenRouter SDKで生成する

`samples/`には、OpenRouter SDKから4種類の生成APIを呼び出すサンプルがあります。
各サンプルは異なる開発元のモデルを使います。

| 生成内容 | モデル | コマンド | 出力 |
| --- | --- | --- | --- |
| テキスト | `deepseek/deepseek-v4-flash-0731` | `bun run sample:text` | `outputs/text.txt` |
| 画像 | `krea/krea-2-medium` | `bun run sample:image` | `outputs/image.png` |
| 動画 | `bytedance/seedance-2.0-fast` | `bun run sample:video` | `outputs/video.mp4` |
| 音声 | `qwen/qwen-audio-3.0-tts-flash` | `bun run sample:tts` | `outputs/speech.mp3` |

### テキストを生成する

次のコマンドはDeepSeek V4 Flash 0731へ日本語の指示を送り、生成されたテキストを`outputs/text.txt`へ保存します。

```console
bun run sample:text
```

### 画像を生成する

次のコマンドはKrea 2 Mediumで、1つのアプリから複数のAIモデルへリクエストを振り分けるOpenRouterのコンセプト画像を生成します。
生成された画像は`outputs/image.png`へ保存されます。

```console
bun run sample:image
```

### 動画を生成する

次のコマンドはSeedance 2.0 Fastで480p、1:1、4秒の動画を生成します。
動画生成は非同期で実行されるため、コマンドは完了まで状態を確認してからMP4ファイルを保存します。

```console
bun run sample:video
```

動画の生成には数分かかる場合があります。

### 音声を生成する

次のコマンドはQwen-Audio 3.0 TTS Flashで日本語の短文を読み上げます。
このサンプルはOpenRouterが公開する音声ID`loongjohn`を使います。

```console
bun run sample:tts
```

`outputs/`はGitの管理対象から除外しています。

## AIエージェントをOpenRouterで動かす

記事では単純にOpenRouterを指定した状態でDeep AgentsやClaude Codeを起動する例を示しましたが、本リポジトリではコマンドベースで動作を確認できるようにしています。

モデルには`deepseek/deepseek-v4-flash-0731`を指定しています。

Deep AgentsとClaude Codeには、税込価格を計算する同じコードの修正を依頼します。
現在のコードは、単価へ税率を適用して小数部分を切り捨てたあとに数量を掛けています。
単価と数量から求めた小計へ税率を適用し、最後に小数部分を切り捨てるように修正する課題です。
どちらのコマンドも`workspace/`を個別の作業ディレクトリへコピーするため、元のコードは変更しません。

### Deep Agentsで修正する

次のコマンドは`workspace/`を`.workspaces/deepagents/`へコピーし、コピーしたコードだけを編集します。

```console
bun run deepagents
```

コマンドが終了すると、Deep Agentsの応答とテスト結果が表示されます。

### Claude Codeで修正する

次のコマンドは`workspace/`を`.workspaces/claude-code/`へコピーし、Claude Codeを非対話で実行します。

```console
bun run claude
```

コマンドは最後にテストを実行し、2件成功すれば終了します。

## 選ばれた推論プロバイダを確認する

次のコマンドは小さなリクエストを1回送り、ルーターが検討した推論プロバイダ、実際に応答した推論プロバイダ、フォールバックの試行履歴を表示します。

```console
bun run route
```

モデル、選択した設定、各候補の`selected`、試行回数、試行履歴がJSONで表示されます。

## 推論プロバイダの選び方を変える

`OPENROUTER_ROUTING_PROFILE`には`default`、`price`、`latency`、`throughput`のいずれかを指定します。

```console
OPENROUTER_ROUTING_PROFILE=price bun run route
OPENROUTER_ROUTING_PROFILE=latency bun run route
```

同じ環境変数はDeep Agentsの実行にも適用できます。

```console
OPENROUTER_ROUTING_PROFILE=price bun run deepagents
```
