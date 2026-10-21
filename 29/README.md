# Software Design誌「実践LLMアプリケーション開発」第29回サンプルコード

LangChain v1.0における`create_agent`とミドルウェアの仕組みを実際に動かして理解するJupyter Notebook

## 証明書検証に関する更新（2026年10月22日）

[eMudhraの中間CA侵害](https://emudhra.com/en/post/certificate-security-incident)への対応として、Pythonを紙面に記載した3.12から3.13.16へ更新しました。
HTTPS通信に使用するCA証明書も、certifi 2026.10.19へ更新しています。

環境構築にはuv 0.13.4以降を使用してください。
以下の手順で`uv sync`を実行した後、Jupyter NotebookまたはJupyter Labを起動するシェルで次の環境変数を設定してください。

```bash
export SSL_CERT_FILE="$(uv run python -m certifi)"
export SSL_CERT_DIR="$(dirname "$SSL_CERT_FILE")"
```

## セットアップ方法

### 環境構築の手順

; このサンプルは`uv`を使用しています。`uv`をインストールしていない場合は[公式サイト](https://github.com/astral-sh/uv)を参照してください。

コマンドを実行して依存関係をインストールしてください。

```
$ uv sync
```

次にサンプルコードを実行するため`.env`ファイルを作成しAPI鍵を設定してください。

```
$ vi .env # 好きなエディタで編集してください
```

`.env`ファイルには以下のAPI鍵を設定してください。

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

- `ANTHROPIC_API_KEY`: Anthropic Claude APIの鍵

### Notebookの実行

Jupyter Notebookを起動します:

```bash
uv run jupyter notebook create_agent.ipynb
```

または、Jupyter Labを使う場合:

```bash
uv run jupyter lab create_agent.ipynb
```

## 内容

`create_agent.ipynb`では以下の内容を解説しています

### 基本編

| セクション | 内容 |
|:---:|:---|
| 1 | 環境セットアップ |
| 2 | create_agentの基本 - 最もシンプルなエージェント作成 |
| 3 | toolsパラメータ - @toolデコレータを使ったツール定義 |
| 4 | modelパラメータ - 文字列/インスタンスでの指定 |
| 5 | system_promptパラメータ - SystemMessageの活用 |
| 6 | response_formatパラメータ - 構造化出力 (ToolStrategy) |

### 応用編

| セクション | 内容 |
|:---:|:---|
| 7 | 短期記憶 (checkpointer) - InMemorySaverを使った会話管理 |
| 8 | ToolRuntimeの活用 - state/context/storeへのアクセス |
| 9 | 長期記憶 (store) - InMemoryStoreを使った永続化 |
| 10 | state_schemaとcontext_schema - カスタム状態とコンテキスト定義 |

### ミドルウェア編

| セクション | 内容 |
|:---:|:---|
| 11 | デコレータベースミドルウェア - @before_model, @after_model, @dynamic_prompt |
| 12 | クラスベースミドルウェア - AgentMiddlewareの継承 |
| 13 | ModelRequestとModelResponse - wrap_model_callを使った詳細制御 |
| 14 | 標準ミドルウェアの紹介 - SummarizationMiddleware等の実例 |

## 注意

- このNotebookを実行するとAnthropic APIが呼び出され料金が発生します
- 実行は自己責任でお願いします
- APIキーは`.env`ファイルで管理し、絶対に公開しないでください
