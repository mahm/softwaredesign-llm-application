# Software Design誌「実践LLMアプリケーション開発」第20回サンプルコード

## サンプルコードの実行方法

### プロジェクトのセットアップ

※ このプロジェクトは`uv`を使用しています。`uv`のインストール方法については[こちら](https://github.com/astral-sh/uv)をご確認ください。

以下のコマンドを実行し、必要なライブラリのインストールを行って下さい。

```
$ uv sync
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
