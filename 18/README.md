# LangGraph Studio プロジェクト

このプロジェクトは、LangGraph Studioに対応したLangGraphソースコードのプロジェクトです。
複数のAIエージェントが協力して、データの検索、データセット生成、チャート生成を行います。

## セットアップ手順

### 1. 必要なパッケージのインストール

```bash
# Pythonの仮想環境を作成
python -m venv .venv
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 日本語フォントのインストール

Ubuntuの場合：
```bash
sudo apt-get update
sudo apt-get install -y fonts-ipafont-gothic fonts-noto-cjk
```

macOSの場合：
```bash
brew install font-ipa
```

### 3. 環境変数の設定

1. `.env.sample`ファイルを`.env`にコピー
```bash
cp .env.sample .env
```

2. `.env`ファイルを編集し、以下のAPIキーを設定
- `ANTHROPIC_API_KEY`: Claude APIのキー
- `TAVILY_API_KEY`: Tavily Search APIのキー
- `LANGSMITH_API_KEY`: LangSmith APIのキー（オプション）
- `LANGSMITH_PROJECT`: LangSmithプロジェクト名（オプション）

## 使用方法

```bash
python -m sd_18.agent
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
