# Software Design誌「実践LLMアプリケーション開発」第24回サンプルコード

## 証明書検証に関する更新（2026年10月22日）

[eMudhraの中間CA侵害](https://emudhra.com/en/post/certificate-security-incident)への対応として、Pythonを紙面に記載した3.12から3.12.15へ更新しました。
HTTPS通信に使用するCA証明書も、certifi 2026.10.19へ更新しています。

環境構築にはuv 0.13.4以降を使用してください。
以下の手順で`uv sync`を実行した後、サンプルを起動するシェルで次の環境変数を設定してください。

```bash
export SSL_CERT_FILE="$(uv run python -m certifi)"
export SSL_CERT_DIR="$(dirname "$SSL_CERT_FILE")"
```

## サンプルコードの実行方法

### プロジェクトのセットアップ

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

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=sd-24
```

- `ANTHROPIC_API_KEY`: Claude APIのキー
- `TAVILY_API_KEY`: Web検索機能用のAPIキー
- `LANGCHAIN_API_KEY`: LangSmithのAPIキー（オプション）

### 実行方法

LangGraph Studioでの実行:

```bash
uv run langgraph dev
```

コマンドラインでの実行:

```bash
# 対話形式で実行
uv run python main.py

# 直接質問を指定して実行
uv run python main.py "LangChainについて教えて"

# デバッグモードで実行
uv run python main.py "2025年のAI動向をレポートして" --debug
```

### コマンドライン引数

- `query`: エージェントへの質問や指示（省略時は対話モード）
- `--debug`: デバッグモードで実行（詳細なイベント情報を表示）
