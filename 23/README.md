# Software Design誌「実践LLMアプリケーション開発」第23回サンプルコード

## 概要

このプロジェクトは、LangGraphのSupervisorパターンとSwarmパターンのシンプルなサンプル実装です。

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
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=sd-23
```

- `ANTHROPIC_API_KEY`: Claude APIのキー
- `LANGCHAIN_API_KEY`: LangSmithのAPIキー（オプション）

### 実行方法

LangGraph Studioでの実行:

```bash
uv run langgraph dev
```

コマンドラインでの実行:

```bash
# 両方のパターンを実行
uv run python main.py

# Supervisorパターンのみ実行
uv run python main.py supervisor

# Swarmパターンのみ実行
uv run python main.py swarm
```

## グラフの説明

### Supervisorパターン (`supervisor`)

- **数学エージェント**: 数学的な計算（加算、乗算、除算）を処理
- **調査エージェント**: 情報収集と調査タスクを処理（モック実装）
- **Supervisor**: タスクを適切なエージェントに委譲

使用例:
- "10と20を足して、その結果に5を掛けてください"
- "人工知能について調べて、その後2の10乗を計算してください"

### Swarmパターン (`swarm`)

- **FAQサポート**: よくある質問に対応
- **技術サポート**: 技術的な問題を処理
- エージェント間で動的にハンドオフ（制御の受け渡し）が可能

使用例:
- "パスワードをリセットしたいです"（FAQ対応）
- "システムがエラーを表示しています"（技術サポートへハンドオフ）

## プロジェクト構造

```
src/sd_23/
├── agents/
│   ├── math_agent.py       # 数学計算エージェント
│   ├── research_agent.py   # 調査エージェント
│   ├── faq_agent.py        # FAQサポートエージェント
│   └── tech_agent.py       # 技術サポートエージェント
├── supervisor_graph.py     # Supervisorパターンのグラフ
└── swarm_graph.py         # Swarmパターンのグラフ
```

## 使用モデル

すべてのエージェントとSupervisorはAnthropic Claude 4 Sonnetを使用しています。

## main.pyの詳細

`main.py`は、両方のパターンの動作確認用スクリプトです。

- **Supervisorパターンのテストケース**: AI技術の市場調査と成長率計算
  - research_expertが最新のAI技術と市場規模を調査
  - math_expertが5年後の市場規模を計算
  
- **Swarmパターンのテストケース**: エラーコード500の診断
  - faq_supportが初期対応
  - tech_supportが詳細な診断と解決策を提供

各エージェントの動作やツール呼び出しがリアルタイムで表示され、マルチエージェントシステムの協調動作を確認できます。
