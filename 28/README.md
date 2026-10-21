# Software Design誌「実践LLMアプリケーション開発」第28回サンプルコード

このサンプルコードでは、DSPy ReActベースのファイル探索エージェントをGEPA（Gradient-Estimation with Prompt Augmentation）により自動最適化します。

## 証明書検証に関する更新（2026年10月22日）

[eMudhraの中間CA侵害](https://emudhra.com/en/post/certificate-security-incident)への対応として、Pythonを紙面に記載した3.13から3.13.16へ更新しました。
HTTPS通信に使用するCA証明書も、certifi 2026.10.19へ更新しています。

環境構築にはuv 0.13.4以降を使用してください。
以下の手順で`uv sync`を実行した後、サンプルを起動するシェルで次の環境変数を設定してください。

```bash
export SSL_CERT_FILE="$(uv run python -m certifi)"
export SSL_CERT_DIR="$(dirname "$SSL_CERT_FILE")"
```

## Dev Container環境の起動

このサンプルコードは、Dev Container環境での実行を推奨します。

1. VS Codeでプロジェクトのルートディレクトリを開く
2. コマンドパレット（Cmd/Ctrl+Shift+P）から「Dev Containers: Reopen in Container」を選択
3. コンテナのビルドと起動が完了するまで待機

## 前提条件

このサンプルコードを実行するには、事前に**OpenAI APIキー**の取得が必要です。
APIキーは[OpenAI APIキーの管理画面](https://platform.openai.com/api-keys)から取得できます。

※ このリポジトリには最適化済みのモデルファイル（`artifact/agent_gepa_optimized_latest.json`）が含まれているため、すぐに評価を試すことができます。

## セットアップ

Dev Container内で以下のコマンドを実行してください。

### 1. 依存関係のインストール

```bash
cd 28
uv sync
```

### 2. 環境変数の設定

`.env.sample`をコピーして`.env`ファイルを作成し、APIキーを設定してください。

```bash
cp .env.sample .env
vi .env  # お好きなエディタで編集してください
```

#### OpenAI APIを使用する場合

```
PROVIDER_NAME=openai
OPENAI_API_KEY=your_openai_api_key_here
SMART_MODEL=gpt-4.1
FAST_MODEL=gpt-4.1-nano
EVAL_MODEL=gpt-4.1-mini
```

#### Azure OpenAI Serviceを使用する場合

```
PROVIDER_NAME=azure
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2025-04-01-preview
SMART_MODEL=gpt-4.1
FAST_MODEL=gpt-4.1-nano
EVAL_MODEL=gpt-4.1-mini
```

## 実行方法

### 1. 評価の実行（推奨）

ベースラインと最適化済みモデルの性能をテストセット（5例）で比較します。

```bash
uv run python agent_evaluation.py
```

出力例：
```
============================================================
FINAL COMPARISON
============================================================
Baseline Average Score:  0.620
Optimized Average Score: 0.750
Improvement:             +0.130
============================================================
```

### 2. エージェントの最適化（オプション）

GEPAを使用してエージェントを最適化します。最適化には約3時間かかります。

```bash
# デフォルトのシード値（42）で実行
uv run python agent_optimization_gepa.py

# シード値を指定して実行
uv run python agent_optimization_gepa.py --seed 123
```

最適化済みモデルは`artifact/agent_gepa_optimized_YYYYMMDD_HHMM_scoreXXX.json`に保存されます。

### 3. エージェントの単独実行（オプション）

最適化済みエージェントを使って任意のタスクを実行できます：

```bash
# ディレクトリ構造を分析
uv run python main.py --task "Analyze the directory structure and create a report" --directory ../27

# Pythonファイルを探索
uv run python main.py --task "Find all Python files and count lines of code" --directory .

# 反復回数を制限
uv run python main.py --task "List main files" --directory . --max-iters 5
```

## プロジェクト構成

### コアモジュール
- `config.py`: 環境変数設定とLLMモデルの初期化
- `agent_module.py`: DSPy ReActベースのファイル探索エージェント実装
- `dataset_loader.py`: ファイル探索タスクのデータセット読み込み

### スクリプト
- `agent_evaluation.py`: ベースラインと最適化モデルの比較スクリプト
- `agent_optimization_gepa.py`: GEPA最適化スクリプト
- `main.py`: エージェント実行用CLIエントリポイント

### データとアーティファクト
- `artifact/`: 最適化済みモデルの保存先
- `logs/`: 最適化実行ログ
- `tmp/reports/`: 評価レポート
