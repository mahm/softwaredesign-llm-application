# Software Design誌「実践LLMアプリケーション開発」第33回サンプルコード

Agent Client Protocol（ACP）を使って、Deep AgentsベースのコーディングアシスタントをZedエディタと接続するサンプルです。

## サンプルコードの実行方法

### 前提条件

- [mise](https://mise.jdx.dev/)（ランタイムバージョン管理）
- [Zed Editor](https://zed.dev/) v0.231.2以上
- Anthropic APIキー

### プロジェクトのセットアップ

```bash
mise trust
mise install
mise run install
```

次に環境変数を設定します。

```bash
cp .env.sample .env
vi .env
```

`.env`ファイルを編集し、Anthropic APIキーを設定してください。

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Zedエディタの設定

Zedのコマンドパレット（Cmd+Shift+P）から「agent: open settings」を実行します。「External Agents」セクションで「+ Add Agent」を選び、`settings.json`に次の設定を追加してください。

```json
{
  "agent_servers": {
    "Deep Agents": {
      "type": "custom",
      "command": "bun",
      "args": [
        "run",
        "/absolute/path/to/softwaredesign-llm-application/33/server.ts"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

- `/absolute/path/to/`の部分はクローン先の実際の絶対パスに置き換えてください。ACPの仕様上、相対パスでは動作しません。
- `ANTHROPIC_API_KEY`にはAnthropicのAPIキーを設定してください。Zedはサブプロセスとしてサーバーを起動するため、`33/.env`ファイルは読み込まれません。Zedの設定で直接指定する必要があります。

### 接続確認

Zedでプロジェクトを開き、エージェントパネルから「New Deep Agents Thread」を選択してスレッドを開始します。テキストを入力して送信すると、エージェントが応答します。

## 参考リンク

- [deepagents-acp](https://github.com/langchain-ai/deepagentsjs/tree/main/libs/acp) - Deep AgentsのACP連携パッケージ
- [ACP公式サイト](https://agentclientprotocol.com/) - Agent Client Protocolの仕様
- [Zed External Agents](https://zed.dev/docs/ai/external-agents) - Zedの外部エージェント設定
