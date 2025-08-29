# Software Design誌「実践LLMアプリケーション開発」第25回サンプルコード

## サンプルコードの実行方法

### 前提条件

このサンプルコードを実行するには、事前に**OpenAI APIキー**の取得が必要です。
APIキーは[OpenAI APIキーの管理画面](https://platform.openai.com/api-keys)から取得できます。

※ このリポジトリには最適化済みのモデルファイル（`artifact/edamame_fairy_model.json`）が含まれているため、すぐにチャットボットを試すことができます。独自に最適化を行いたい場合は、後述の「チャットボットの最適化」手順を実行してください。

### プロジェクトのセットアップ

※ このプロジェクトは`uv`を使用しています。`uv`のインストール方法については[こちら](https://github.com/astral-sh/uv)をご確認ください。

以下のコマンドを実行し、必要なライブラリのインストールを行って下さい。

```
$ uv sync
```

次に環境変数の設定を行います。`.env`ファイルを作成し、以下の内容を記載してください。

```
$ vi .env # お好きなエディタで編集してください
```

`.env`ファイルに以下の設定を記載してください。

```
OPENAI_API_KEY=your_openai_api_key_here
MLFLOW_PORT=5000
```

- `OPENAI_API_KEY`: OpenAI APIのキー（gpt-4.1-mini, gpt-4.1-nano使用）
- `MLFLOW_PORT`: MLflowサーバーのポート番号（デフォルト: 5000）

### 実行方法

#### 1. MLflowサーバーの起動

実験管理のためのMLflowサーバーを起動します：

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
```

起動後、ブラウザで `http://localhost:5001` にアクセスすると、MLflow UIで実験結果を確認できます。

#### 2. チャットボットの最適化（オプション）

自身でMIPROv2を使用してチャットボットを最適化する場合：

```bash
uv run python chatbot_tuning.py
```

このコマンドで以下の処理が実行されます：
- トレーニングに使用する日本語データセットの読み込み
- MIPROv2による自動プロンプト最適化
- MLflowでの実験追跡
- 最適化済みモデルの保存（`artifact/edamame_fairy_model.json`を上書き）

#### 3. 対話型チャットボットの実行

最適化済みのチャットボットと対話します：

```bash
uv run python main.py
```

- 「quit」「exit」「終了」のいずれかを入力すると終了します
- 枝豆の妖精キャラクターとして、語尾に「のだ」「なのだ」を使った親しみやすい対話を行います

### プロジェクト構成

- `chatbot_module.py`: DSPyチャットボットモジュール
- `chatbot_tuning.py`: MIPROv2による最適化スクリプト
- `main.py`: DSPyモジュールのテスト用チャット
- `artifact/`: 最適化済みモデルの保存先
- `mlartifacts/`: MLflow実験のアーティファクト（MLflow実行時に自動生成）
- `mlflow.db`: MLflow実験データベース（MLflow実行時に自動生成）
