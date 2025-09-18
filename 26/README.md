# Software Design誌「実践LLMアプリケーション開発」第26回サンプルコード

このサンプルコードでは、日本語QAデータセット「JQaRA」を使用してRAGパイプラインの最適化を行います。

## 使用データセット

本サンプルコードでは[JQaRA (Japanese Question Answering with Retrieval Augmentation)](https://huggingface.co/datasets/hotchpotch/JQaRA)データセットを使用しています。JQaRAは日本語の質問応答タスク用に構築された高品質なデータセットで、RAGシステムの評価に適しています。データセット作成者の[hotchpotch](https://huggingface.co/hotchpotch)氏に感謝いたします。

## サンプルコードの実行方法

### 前提条件

このサンプルコードを実行するには、事前に**OpenAI APIキー**の取得が必要です。
APIキーは[OpenAI APIキーの管理画面](https://platform.openai.com/api-keys)から取得できます。

※ このリポジトリには最適化済みのモデルファイル（`artifact/rag_optimized_latest.json`）が含まれているため、すぐに評価を試すことができます。独自に最適化を行いたい場合は、後述の「RAGパイプラインの最適化」手順を実行してください。

### プロジェクトのセットアップ

※ このプロジェクトは`uv`を使用しています。`uv`のインストール方法については[こちら](https://github.com/astral-sh/uv)をご確認ください。

以下のコマンドを実行し、必要なライブラリのインストールを行って下さい。

```
$ uv sync
```

次に環境変数の設定を行います。`.env.sample`をコピーして`.env`ファイルを作成し、以下の内容を記載してください。

```
$ cp .env.sample .env
$ vi .env # お好きなエディタで編集してください
```

`.env`ファイルに以下の設定を記載してください。

#### OpenAI APIを使用する場合

```
PROVIDER_NAME=openai
OPENAI_API_KEY=your_openai_api_key_here
SMART_MODEL=gpt-4.1
FAST_MODEL=gpt-4.1-nano
```

#### Azure OpenAI Serviceを使用する場合

```
PROVIDER_NAME=azure
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2025-04-01-preview
SMART_MODEL=gpt-4.1
FAST_MODEL=gpt-4.1-nano
```

- `SMART_MODEL`: MIPROv2最適化時に使用する高性能モデル
- `FAST_MODEL`: 推論時に使用する高速モデル

### 実行方法

#### 1. RAGパイプラインの最適化

MIPROv2を使用してRAGパイプラインを最適化します。

```bash
uv run python rag_optimization.py
```

このコマンドで以下の処理が実行されます。

- JQaRAデータセット（dev）から50質問を読み込み
- Train/Val分割（50:50）で訓練・検証
- MIPROv2による自動プロンプト最適化
- テストセット（30質問）での評価
- 最適化済みモデルの保存（`artifact/rag_optimized_YYYYMMDD_HHMM_emXXX.json`）

なぜTrain/Valの分割を1:1で行っているかについては、雑誌記事にて解説しています。
実行には約10-15分かかります（APIコール数により変動）。最適化処理には実際にAPI使用料金が発生しますので、注意してください。心配な場合は、ソースコード内の質問数のパラメータを変更してから、お試しください。

#### 2. 評価・比較の実行

ベースラインと最適化済みモデルの性能を比較します：

```bash
uv run python rag_evaluation.py
```

出力例：
```
============================================================
🔬 RAG評価
[Baseline]  EM: 26.7%
[Optimized] EM: 86.7% (Δ +60.0%)
============================================================
```

### プロジェクト構成

- `config.py`: 環境変数設定とLLM/埋め込みモデルの設定
- `dataset_loader.py`: JQaRAデータセット読み込みモジュール
- `evaluator.py`: 共通評価モジュール（dspy.Evaluateを使用）
- `rag_module.py`: RAGパイプライン実装（RewriteQuery、GenerateAnswer）
- `rag_optimization.py`: MIPROv2による最適化スクリプト
- `rag_evaluation.py`: ベースラインと最適化モデルの比較スクリプト
- `artifact/`: 最適化済みモデルの保存先
- `.env.sample`: 環境変数のテンプレート

### 技術概要

このサンプルでは、DSPyフレームワークとMIPROv2最適化を使用して、日本語RAGパイプラインの性能向上を実現しています。

1. **検索クエリ最適化（RewriteQuery）**: ユーザーの質問を検索に適した形にリライト
2. **回答生成（GenerateAnswer）**: 検索結果と質問から回答を生成（Chain of Thought）
3. **MIPROv2最適化**: プロンプトと少数ショット例の自動最適化

JQaRAデータセットを使用し、Exact Match（完全一致）メトリクスで評価を行います。