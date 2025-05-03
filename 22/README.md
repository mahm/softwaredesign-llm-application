# 領収書OCRエージェント

領収書をアップロードまたはカメラで撮影し、OCR処理を行い、勘定科目情報を提案するエージェントシステムです。ユーザーのフィードバックを取り入れながら情報を修正し、最終的にCSVに保存します。

## 機能

- 領収書画像のアップロードまたはカメラ撮影
- Claude Vision APIを使用したOCR処理
- 抽出された情報から勘定科目の自動提案
- ユーザーフィードバックを取り入れた情報修正
- 承認済みデータのCSV保存
- 履歴閲覧とCSVダウンロード

## 技術スタック

- **言語**: Python 3.12+
- **フレームワーク**:
  - LangGraph（タスクフロー管理）
  - Streamlit（UI）
  - LangChain（LLM連携）
- **LLM**: Anthropic Claude (claude-3-5-sonnet-20240620)
- **ストレージ**: CSV（tmp/db.csvに保存）

## インストール手順

### 前提条件

- Python 3.12以上
- Anthropic API キー

### 環境のセットアップ

1. このリポジトリをクローン
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. 仮想環境を作成して有効化
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linuxの場合
   # または
   venv\Scripts\activate  # Windowsの場合
   ```

3. 依存関係をインストール
   ```bash
   pip install -e .
   ```

4. 環境変数を設定
   - `.env`ファイルを作成し、以下の内容を設定:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## 使用方法

1. アプリケーションを実行
   ```bash
   python run.py
   ```

2. ブラウザで `http://localhost:8501` にアクセス

3. 操作手順
   - ファイルアップロードまたはカメラで領収書を撮影
   - 「処理開始」ボタンをクリック
   - OCR処理と勘定科目提案が自動的に行われる
   - 提案された情報を確認し、必要に応じて修正
   - 「この情報で承認する」または「更新して再確認」ボタンをクリック
   - 保存完了後、新たな処理や履歴表示が可能

## ディレクトリ構成

```
/
├── src/
│   ├── receipt_processor/
│   │   ├── __init__.py
│   │   ├── app.py         # Streamlitアプリケーション
│   │   ├── agent.py       # LangGraph Workflow定義
│   │   ├── models.py      # データモデル
│   │   ├── vision.py      # OCR処理
│   │   ├── account.py     # 勘定科目提案
│   │   ├── storage.py     # CSV保存
│   │   └── ui_components.py  # UI関連コンポーネント
│   └── main.py            # エントリーポイント
├── tmp/
│   └── db.csv             # 保存先CSV（自動生成）
├── pyproject.toml         # 依存関係定義
└── run.py                 # 実行スクリプト
```

## 環境変数

- `ANTHROPIC_API_KEY` (必須): Anthropic API キー
- `MODEL_NAME` (オプション): 使用するClaudeモデル名（デフォルト: "claude-3-5-sonnet-20240620"）
- `LOG_LEVEL` (オプション): ログレベル設定
- `CSV_PATH` (オプション): CSV保存先パス（デフォルト: "tmp/db.csv"）

## 注意事項

- 一時的な画像ファイルは処理後に自動的に削除されます
- CSVファイルは更新前に自動的にバックアップが作成されます
- APIキーは`.env`ファイルで管理し、リポジトリにはコミットしないでください
