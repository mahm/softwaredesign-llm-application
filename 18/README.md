# Software Design誌「実践LLMアプリケーション開発」第18回サンプルコード

## 証明書検証に関する更新（2026年10月22日）

[eMudhraの中間CA侵害](https://emudhra.com/en/post/certificate-security-incident)への対応として、Pythonを紙面に記載した3.12.7から3.12.15へ更新しました。
HTTPS通信に使用するCA証明書も、certifi 2026.10.19へ更新しています。

環境構築にはuv 0.13.4以降を使用してください。
以下の手順で`uv sync`を実行した後、サンプルを起動するシェルで次の環境変数を設定してください。

```bash
export SSL_CERT_FILE="$(uv run python -m certifi)"
export SSL_CERT_DIR="$(dirname "$SSL_CERT_FILE")"
```

## サンプルコードの実行方法

### 1. 日本語フォントのインストール

Ubuntuの場合：
```bash
sudo apt-get update
sudo apt-get install -y fonts-ipafont-gothic fonts-noto-cjk
```

macOSの場合：
```bash
brew install font-ipa
```

### 2. プロジェクトのセットアップ

※ このプロジェクトは`uv`を使用しています。`uv`のインストール方法については[こちら](https://github.com/astral-sh/uv)をご確認ください。

以下のコマンドを実行し、必要なライブラリのインストールを行って下さい。

```
$ uv sync
```

次に環境変数の設定を行います。まず`.env.sample`ファイルをコピーして`.env`ファイルを作成します。

```
$ cp .env.sample .env
$ vi .env # お好きなエディタで編集してください
```

`.env`ファイルを編集し、以下のAPIキーを設定してください。

- `ANTHROPIC_API_KEY`: Claude APIのキー
- `TAVILY_API_KEY`: Tavily Search APIのキー
- `LANGSMITH_API_KEY`: LangSmith APIのキー（オプション）
- `LANGSMITH_PROJECT`: LangSmithプロジェクト名（オプション）

## 実行方法

```bash
uv run python -m sd_18.agent
```

## プロジェクト構成

- `sd_18/agent.py`: メインのエージェントコード
- `output/`: 生成されたファイルの保存先（gitignore対象）
  - `{TIMESTAMP}/`: 実行時のタイムスタンプ（YYYYMMDD_HHMMSS形式）
    - `charts/`: 生成されたチャートの保存先
    - `data/`: 生成されたデータセットの保存先

## 注意事項

- チャートの生成には日本語フォントが必要です
- 生成されたファイルは`output/{TIMESTAMP}`ディレクトリ以下に保存されます
  - チャート: `output/{TIMESTAMP}/charts/`
  - データセット: `output/{TIMESTAMP}/data/`
- 各実行結果は実行時のタイムスタンプ付きディレクトリで管理されます
- APIキーは`.env`ファイルで管理し、Gitにコミットしないようにしてください
