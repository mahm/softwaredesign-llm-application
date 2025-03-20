# Software Design誌「実践LLMアプリケーション開発」第20回サンプルコード

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

続けて.envファイル内の以下のキーを設定して下さい。`TAVILY_API_KEY`ならびに`LANGSMITH_API_KEY`の取得方法については次節で解説します。

```
ANTHROPIC_API_KEY=[発行されたAPIキー]
TAVILY_API_KEY=[発行されたAPIキー]
LANGSMITH_API_KEY=[発行されたAPIキー]
```

## 実行方法

以下のコマンドを実行することで、エージェントを実行できます。
MCPサーバは自動的に起動されるため、事前に別途起動する必要はありません。

```bash
# エージェントの実行（コマンドライン引数でエージェントへの依頼を設定）
uv run -m src.sd_20 "LangChainの最新情報を教えてください"
```

また、LangGraph Studioを利用して動作確認をすることもできます。
LangGraph Studioを利用する場合は、次のコマンドでサーバを立ち上げてください。

```bash
uv run langgraph dev
```

## mcp_config.jsonの設定

このサンプルコードでは、MCPサーバの設定を`mcp_config.json`ファイルで管理しています。デフォルトでは以下のような設定になっています：

```json
{
  "mcpServers": {
    "knowledge-db": {
      "command": "uv",
      "args": ["run", "-m", "src.mcp_servers.server"]
    }
  }
}
```

## サンプルコードの内容

本サンプルコードでは、MCPサーバとLangGraphエージェント（`create_react_agent`）との連携を実装しています。

主要なコンポーネント：
- `src/sd_20/mcp_manager.py`: MCPサーバからツールをロードし、LangGraphエージェントで使えるようにする
- `src/sd_20/agent.py`: `create_react_agent`を使用したエージェントの定義
- `src/mcp_servers/database.py`: SQLiteの操作を行うモジュール
- `src/mcp_servers/server.py`: MCPサーバの実装

## Tavily APIキーの取得方法

### Tavilyについて

Tavilyは、LLMアプリケーションにおけるRAG（Retrieval-Augmented Generation）に特化した検索エンジンです。

通常の検索APIはユーザーのクエリに基づいて検索結果を取得しますが、検索の目的に対して無関係な結果が返ってくることがあります。また、単純なURLやスニペットが返されるため、開発者が関連するコンテンツをスクレイピングしたり、不要な情報をフィルタリングしたりする必要があります。

一方でTavilyは、一回のAPI呼び出しで20以上のサイトを集約し、独自のAIを利用しタスクやクエリ、目的に最も関連する情報源とコンテンツを評価、フィルタリング、ランク付けすることが可能な仕組みになっています。

今回のサンプルコードを実行するためにはTavilyのAPIキーが必要なので、以下の手段でAPIキーを取得してください。

### Tavily APIキーの取得

1. まず https://tavily.com/ にアクセスし、画面右上の「Try it out」をクリックします。
2. するとサインイン画面に遷移するので、そのままサインインを進めて下さい。GoogleアカウントかGitHubアカウントを選択することができます。
3. サインインが完了するとAPIキーが取得できる画面に遷移します。画面中央部の「API Key」欄より、APIキーをコピーしてください。
4. APIキーを`.env`内の`TAVILY_API_KEY`に設定してください。

## LangSmith APIキーの取得方法

LangSmithはLLM（大規模言語モデル）実行のログ分析ツールです。LLMアプリケーションの実行を詳細に追跡し、デバッグや評価を行うための機能を提供します。詳細については https://docs.smith.langchain.com/ をご確認ください。

LangSmithによるトレースを有効にするためには、LangSmithにユーザー登録した上で、APIキーを発行する必要があります。以下の手順を参考にAPIキーを取得してください。

### LangSmith APIキーの取得

1. [サインアップページ](https://smith.langchain.com/)より、LangSmithのユーザー登録を行います。
2. [設定ページ](https://smith.langchain.com/settings)を開きます。
3. API keysタブを開き、「Create API Key」をクリックします。するとAPIキーが発行されますので、APIキーをコピーしてください。
4. APIキーを`.env`内の`LANGCHAIN_API_KEY`に設定してください。
