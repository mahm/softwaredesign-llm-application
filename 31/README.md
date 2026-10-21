# Software Design誌「実践LLMアプリケーション開発」第31回サンプルコード

LangChainのミドルウェア機能を活用した3つのデモアプリケーションです。

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
LANGCHAIN_TRACING_V2=true # ※ LangSmithのAPIキー設定は必須ではありません
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=sd-31
```

- `ANTHROPIC_API_KEY`: Anthropic Claude APIのキー
- `LANGCHAIN_API_KEY`: LangSmithのAPIキー (オプション)

### 実行方法

Streamlitアプリの起動:

```bash
uv run python run.py
```

ブラウザで http://localhost:8501 を開いてデモを操作できます。

### テスト実行

```bash
uv run pytest tests/ -v
```

## シナリオ

### シナリオ1: メール処理エージェント

個人情報のフィルタリング、会話履歴の要約、Human in the loopフローのデモ。

**使用ミドルウェア:**
- `PIIMiddleware`: メールアドレス・電話番号のマスキング/ブロック
- `SummarizationMiddleware`: 長い履歴の自動要約
- `HumanInTheLoopMiddleware`: メール送信前の承認

**テスト方法:**
1. 「メール一覧を見せて」でメール一覧を確認
2. 「メール001を読んで」でメール内容を確認
3. 電話番号を含むメッセージを入力してPIIブロックを確認
4. 「田中さんに返信して」で承認フローを確認

### シナリオ2: レジリエンスエージェント

無限ループ防止、コスト制御、エラー耐性のデモ。

**使用ミドルウェア:**
- `ModelCallLimitMiddleware`: モデル呼び出し回数制限
- `ToolCallLimitMiddleware`: ツール呼び出し回数制限
- `ModelFallbackMiddleware`: モデルフォールバック
- `ToolRetryMiddleware` / `ModelRetryMiddleware`: 自動リトライ

**テスト方法:**
1. 「Pythonについて検索して」でWeb検索を実行
2. 実行ログでリトライの様子を確認
3. メトリクスで成功率を確認

### シナリオ3: ツール選択エージェント

LLMによる動的ツール選択のデモ。

**使用ミドルウェア:**
- `LLMToolSelectorMiddleware`: 12個のツールから最適な3つを自動選択

**テスト方法:**
1. 「東京の天気を教えて」でweatherツール選択を確認
2. 「100ドルは何円?」でcurrency_convertツール選択を確認
3. 質問ごとに選択ツールが変化することを確認

## ファイル構成

```
31/
├── src/sd_31/
│   ├── app.py                        # Streamlit エントリポイント
│   ├── agents/
│   │   ├── __init__.py               # AgentResponse型 + エクスポート
│   │   ├── email_agent.py            # シナリオ1: メール処理
│   │   ├── resilient_agent.py        # シナリオ2: レジリエンス
│   │   └── tool_selector_agent.py    # シナリオ3: ツール選択
│   ├── controllers/
│   │   ├── __init__.py               # エクスポート
│   │   └── base.py                   # エージェント対話処理
│   ├── pages/
│   │   ├── common.py                 # 共通ユーティリティ
│   │   ├── scenario1.py              # シナリオ1 UI
│   │   ├── scenario2.py              # シナリオ2 UI
│   │   └── scenario3.py              # シナリオ3 UI
│   └── mock/
│       ├── email_agent.json          # メールモックデータ
│       ├── resilient_agent.json      # 検索/DBモックデータ
│       └── tool_selector_agent.json  # 各種モックデータ
├── tests/
│   ├── test_email_agent.py
│   ├── test_resilient_agent.py
│   └── test_tool_selector_agent.py
├── run.py
├── pyproject.toml
└── .env.sample
```
