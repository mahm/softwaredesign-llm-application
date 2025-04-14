# LangGraph Content Creator

LangGraph Functional APIを使用して、ユーザーの指示に基づいたコンテンツを生成し、フィードバックを取り入れながら改善していくHuman-in-the-loopのStreamlitアプリケーションです。

## 概要

このアプリケーションは、以下の機能を提供します：

- ユーザーの指示に基づく高品質なコンテンツ生成
- 生成されたコンテンツに対するフィードバック候補の提案
- フィードバックを取り入れたコンテンツの改善
- 直感的なチャットインターフェース
- 生成されたコンテンツのクリップボードへのコピー機能

## プロジェクト構成

```
/
├── src/
│   ├── content_creator/
│   │   ├── __init__.py        # パッケージ初期化
│   │   ├── app.py             # メインアプリケーション
│   │   ├── langraph_workflow.py  # LangGraphワークフロー
│   │   └── ui_components.py   # UIコンポーネント
│   └── main.py               # メインエントリポイント
├── .env.example              # 環境変数設定例
├── pyproject.toml            # プロジェクト設定
└── run.py                    # 実行スクリプト
```

## 技術スタック

- **UI**: Streamlit
- **バックエンド**: Python 3.12
- **ワークフロー管理**: LangGraph
- **LLM統合**: LangChain
- **LLM**: Anthropic Claude
- **環境変数管理**: python-dotenv

## セットアップ

1. 必要な環境変数の設定
   ```
   cp .env.example .env
   ```
   `.env`ファイルを編集し、Anthropic APIキーを設定します：
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## 実行方法

```bash
python run.py
```

または：

```bash
streamlit run src/content_creator/app.py
```

## 使い方

1. 左側のチャット欄に指示を入力します
2. 右側に生成されたコンテンツが表示されます
3. 提案されたフィードバックオプションを選択するか、自分でフィードバックを入力します
4. フィードバックに基づいてコンテンツが改善されます
5. 必要に応じてクリップボードにコピーボタンを使用してコンテンツをコピーできます

## ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。 