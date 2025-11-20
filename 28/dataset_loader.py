# -*- coding: utf-8 -*-
"""
Dataset loader for file exploration agent training and evaluation.

This module provides criteria-based datasets for training and evaluating the file exploration agent.
All tasks require multi-file exploration and are evaluated using explicit criteria with LLM as a Judge.
"""

import dspy
from typing import List


def load_training_dataset() -> List[dspy.Example]:
    """
    Load training dataset for file exploration tasks.

    10 high-quality training examples requiring multi-file exploration:
    - Easy: 2 tasks (2-3 files)
    - Medium: 5 tasks (3-5 files)
    - Hard: 3 tasks (5-8 files)

    Returns:
        List of dspy.Example instances with task, working_directory, and criteria fields
    """
    examples = [
        # ===== Training Task 1: Easy (2 files) =====
        dspy.Example(
            task="config.pyでSMART_MODELとFAST_MODELを見つけ、rag_optimization_gepa.pyでこれらがどのように使われているか（どの変数に代入され、どの関数に渡されるか）を調査せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - config.pyを読んだか: 1点
   - rag_optimization_gepa.pyを読んだか: 1点

2. 必須要素の言及（6点）
   - SMART_MODEL定義（config.py内のSMART_MODEL = os.getenv(...)）: 1.5点
   - FAST_MODEL定義（config.py内のFAST_MODEL = os.getenv(...)）: 1.5点
   - reflection_lm変数への代入（reflection_lm = configure_lm(SMART_MODEL, temperature=1.0)）: 1.5点
   - fast_lm変数への代入（fast_lm = configure_lm(FAST_MODEL, temperature=0.0)）: 1.5点

3. 情報統合（2点）
   - config.pyでの定義 → rag_optimization_gepa.pyでのimport → 使用の流れを説明: 1点
   - temperature設定の違い（1.0 vs 0.0）に言及: 1点

オプション（加点）:
   - README.mdまたはCLAUDE.mdでモデル選択の背景を確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 2: Easy (2-3 files) =====
        dspy.Example(
            task="rag_module.pyの3つのクラス（RewriteQuery、GenerateAnswer、RAGQA）を特定し、これらのクラスがimportしている依存モジュール（dspy.Signature、dspy.Predictなど）を追跡せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（1点）
   - rag_module.pyを読んだか: 1点

2. 必須要素の言及（7点）
   - RewriteQueryクラスの特定: 1点
   - GenerateAnswerクラスの特定: 1点
   - RAGQAクラスの特定: 1点
   - dspy.Signatureの使用（RewriteQuery、GenerateAnswerの基底クラス）: 1.5点
   - dspy.Predictの使用（RAGQAクラス内でのインスタンス化）: 1.5点
   - dspy.Moduleの継承（RAGQAクラスの基底クラス）: 1点

3. 情報統合（2点）
   - 3クラスの役割分担（Signature定義 → Predictorによる実行 → Module統合）を説明: 2点

オプション（加点）:
   - README.mdでRAGパイプライン全体像を確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 3: Medium (3 files) =====
        dspy.Example(
            task="dataset_loader.pyとevaluator.pyを調査し、JQaRAデータセットがどのようにDSPy形式に変換され、どのメトリクス関数で評価されるかを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - dataset_loader.pyを読んだか: 1点
   - evaluator.pyを読んだか: 1点

2. 必須要素の言及（6点）
   - load_jqara_dataset関数の特定: 1点
   - dspy.Example形式への変換処理（.with_inputs("question", "answer")）: 1.5点
   - exact_match_metric関数の特定: 1点
   - rag_comprehensive_metric関数の特定: 1点
   - positive/negativeデータの分離処理: 1.5点

3. 情報統合（2点）
   - データフロー（JQaRA JSON → load_jqara_dataset → dspy.Example → metric評価）を説明: 2点

オプション（加点）:
   - rag_optimization_gepa.pyでメトリクス使用箇所を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 4: Medium (3 files) =====
        dspy.Example(
            task="embeddings_cache.pyのキャッシュ戦略（MD5ハッシュ、pickle保存）を調査し、rag_module.pyまたはrag_optimization_gepa.pyでの使用箇所を特定せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - embeddings_cache.pyを読んだか: 1点
   - rag_module.pyまたはrag_optimization_gepa.pyを読んだか: 1点

2. 必須要素の言及（6点）
   - MD5ハッシュによるキャッシュキー生成: 1.5点
   - pickleによるキャッシュ保存（cache/ディレクトリ）: 1.5点
   - get_cached_embedding関数の特定: 1点
   - save_embedding_cache関数の特定: 1点
   - キャッシュヒット/ミス判定ロジック: 1点

3. 情報統合（2点）
   - キャッシュ戦略の目的（計算コスト削減、API呼び出し削減）を説明: 1点
   - 使用箇所との連携（どのように呼び出されるか）を説明: 1点

オプション（加点）:
   - README.mdでキャッシュ効果（35x効率化）に言及: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 5: Medium (2 files) =====
        dspy.Example(
            task="rag_optimization_gepa.pyのログ機能（Teeクラス、setup_logging、cleanup_logging）を調査し、ログファイルがどこに保存されるかを特定せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（1点）
   - rag_optimization_gepa.pyを読んだか: 1点

2. 必須要素の言及（7点）
   - Teeクラスの特定（二重出力機能）: 1.5点
   - setup_logging関数の特定: 1点
   - cleanup_logging関数の特定: 1点
   - ログファイルパス（logs/gepa_optimization_YYYYMMDD_HHMM.log）: 1.5点
   - 標準出力ログパス（logs/gepa_optimization_YYYYMMDD_HHMM_stdout.log）: 1.5点
   - タイムスタンプ生成（datetime.now().strftime("%Y%m%d_%H%M")）: 0.5点

3. 情報統合（2点）
   - setup → 実行 → cleanup のライフサイクルを説明: 1点
   - 二重出力の仕組み（console + file同時書き込み）を説明: 1点

オプション（加点）:
   - logs/ディレクトリで実際のログファイルを確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 6: Hard (8 files) =====
        dspy.Example(
            task="27ディレクトリのRAGシステム全体像を調査せよ。全8個のPythonファイル（config.py、dataset_loader.py、embeddings_cache.py、evaluator.py、rag_evaluation.py、rag_module.py、rag_optimization.py、rag_optimization_gepa.py）の役割と相互関係を説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（4点）
   - config.pyを読んだか: 0.5点
   - dataset_loader.pyを読んだか: 0.5点
   - embeddings_cache.pyを読んだか: 0.5点
   - evaluator.pyを読んだか: 0.5点
   - rag_evaluation.pyを読んだか: 0.5点
   - rag_module.pyを読んだか: 0.5点
   - rag_optimization.pyを読んだか: 0.5点
   - rag_optimization_gepa.pyを読んだか: 0.5点

2. 必須要素の言及（4点）
   - config.py: 環境設定とモデル管理: 0.5点
   - dataset_loader.py: JQaRAデータ読み込み: 0.5点
   - embeddings_cache.py: 埋め込みキャッシュ: 0.5点
   - evaluator.py: 評価メトリクス: 0.5点
   - rag_evaluation.py: ベースライン vs 最適化比較: 0.5点
   - rag_module.py: RAGパイプライン実装: 0.5点
   - rag_optimization.py: MIPROv2最適化（レガシー）: 0.5点
   - rag_optimization_gepa.py: GEPA最適化: 0.5点

3. 情報統合（2点）
   - アーキテクチャフロー（データ読み込み → RAGパイプライン → 最適化 → 評価）を説明: 1点
   - ファイル間の依存関係（import関係）を説明: 1点

オプション（加点）:
   - README.mdでプロジェクト全体像を確認: +0.5点
   - CLAUDE.mdで実装ガイドを確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 7: Hard (4 files) =====
        dspy.Example(
            task="GEPA最適化フローを調査せよ。config.pyのモデル設定 → rag_optimization_gepa.pyの最適化実行 → rag_module.pyのプロンプト改善 → 評価 のサイクルを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - config.pyを読んだか: 1点
   - rag_optimization_gepa.pyを読んだか: 1点
   - rag_module.pyを読んだか: 1点

2. 必須要素の言及（5点）
   - SMART_MODEL（GEPA reflection用、temperature=1.0）: 1点
   - FAST_MODEL（推論用、temperature=0.0）: 1点
   - dspy.GEPA初期化（metric、reflection_lm、candidate_selection_strategy="pareto"）: 1.5点
   - optimizer.compile実行（trainset、valset）: 1点
   - rag_module.pyのクラス（RewriteQuery、GenerateAnswer）がプロンプト最適化対象: 0.5点

3. 情報統合（2点）
   - 最適化サイクル（評価 → フィードバック → プロンプト改善 → 再評価）を説明: 1点
   - Pareto戦略（複数候補の並行探索）に言及: 1点

オプション（加点）:
   - README.mdでGEPA最適化結果（20%改善、35x効率化）を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 8: Hard (4 files) =====
        dspy.Example(
            task="データ処理フローを調査せよ。dataset_loader.pyでのJQaRA読み込み → rag_module.pyでの処理 → evaluator.pyでの評価 → rag_optimization_gepa.pyでのメトリクス使用 の流れを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - dataset_loader.pyを読んだか: 1点
   - rag_module.pyを読んだか: 1点
   - evaluator.pyを読んだか: 0.5点
   - rag_optimization_gepa.pyを読んだか: 0.5点

2. 必須要素の言及（5点）
   - load_jqara_dataset関数によるJSONデータ読み込み: 1点
   - dspy.Example形式への変換（question、answer、context、is_positive）: 1点
   - RAGQAクラスによる処理（RewriteQuery → GenerateAnswer）: 1.5点
   - exact_match_metricまたはrag_comprehensive_metricによる評価: 1点
   - gepa_metric_with_feedbackでのメトリクス使用: 0.5点

3. 情報統合（2点）
   - データフロー全体（JSON → Example → RAG処理 → 評価）を説明: 1点
   - positive/negativeデータの扱い方に言及: 1点

オプション（加点）:
   - README.mdでJQaRAデータセット詳細を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 9: Medium (5 files) =====
        dspy.Example(
            task="エラーハンドリング実装を調査せよ。config.py、dataset_loader.py、embeddings_cache.py、evaluator.py、rag_optimization_gepa.pyの各ファイルでどのような例外処理が実装されているかを特定せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - config.pyを読んだか: 0.5点
   - dataset_loader.pyを読んだか: 0.5点
   - embeddings_cache.pyを読んだか: 0.5点
   - evaluator.pyを読んだか: 0.5点
   - rag_optimization_gepa.pyを読んだか: 1点

2. 必須要素の言及（5点）
   - config.pyのAPI key検証（KeyError、ValueError処理）: 1点
   - dataset_loader.pyのファイル読み込みエラー処理（FileNotFoundError、JSONDecodeError）: 1点
   - embeddings_cache.pyのキャッシュI/Oエラー処理（pickle.PickleError、OSError）: 1点
   - evaluator.pyの評価エラー処理（AttributeError、ZeroDivisionError）: 1点
   - rag_optimization_gepa.pyのログ処理とcleanup（tryfinally）: 1点

3. 情報統合（2点）
   - エラーハンドリング戦略（早期検出、ユーザーフレンドリーメッセージ）を説明: 1点
   - 各ファイルのエラー処理パターンの共通点を説明: 1点

オプション（加点）:
   - README.mdでトラブルシューティング情報を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Training Task 10: Medium (4 files) =====
        dspy.Example(
            task="プロジェクト設定ファイル群を調査せよ。README.md、CLAUDE.md、pyproject.toml、.envの役割と、これらがPythonコードからどのように参照されるかを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - README.mdを読んだか: 0.5点
   - CLAUDE.mdを読んだか: 0.5点
   - pyproject.tomlを読んだか: 1点
   - .envまたは.env.sampleを読んだか: 1点

2. 必須要素の言及（5点）
   - README.md: プロジェクト概要、主な機能、使用方法: 1点
   - CLAUDE.md: Claude Code向け実装ガイド: 1点
   - pyproject.toml: 依存関係（dspy、datasets、openai）: 1.5点
   - .env: 環境変数（OPENAI_API_KEY、PROVIDER_NAME）: 1.5点

3. 情報統合（2点）
   - config.pyでの.env読み込み（os.getenv）を説明: 1点
   - pyproject.tomlの依存関係がPythonコードでimportされる流れを説明: 1点

オプション（加点）:
   - config.pyで環境変数の実際の使用箇所を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),
    ]

    return examples


def load_test_dataset() -> List[dspy.Example]:
    """
    Load test dataset for file exploration tasks.

    5 high-quality test examples with unseen exploration patterns:
    - Easy: 1 task (2-3 files)
    - Medium: 2 tasks (3-5 files)
    - Hard: 2 tasks (5-8 files)

    Returns:
        List of dspy.Example instances with task, working_directory, and criteria fields
    """
    examples = [
        # ===== Test Task 1: Easy (3 files) =====
        dspy.Example(
            task="config.pyの環境変数（OPENAI_API_KEY、PROVIDER_NAME、AZURE_OPENAI_ENDPOINT）が、rag_optimization_gepa.pyとrag_module.pyでどのように使用されているかを追跡せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - config.pyを読んだか: 1点
   - rag_optimization_gepa.pyを読んだか: 1点
   - rag_module.pyを読んだか: 1点

2. 必須要素の言及（5点）
   - config.pyでのos.getenv呼び出し（OPENAI_API_KEY、PROVIDER_NAME、AZURE_OPENAI_ENDPOINT）: 1.5点
   - configure_lm関数での環境変数使用: 1点
   - rag_optimization_gepa.pyでのconfigure_lm呼び出し（reflection_lm、fast_lm）: 1.5点
   - rag_module.pyまたはrag_optimization_gepa.pyでのdspy.configure(lm=...)呼び出し: 1点

3. 情報統合（2点）
   - 環境変数 → config.py読み込み → configure_lm → LM初期化 の流れを説明: 1点
   - OpenAI vs Azure OpenAIの切り替え方法に言及: 1点

オプション（加点）:
   - .envまたは.env.sampleで環境変数のサンプル値を確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== Test Task 2: Medium (3 files) =====
        dspy.Example(
            task="rag_optimization_gepa.pyのgepa_metric_with_feedbackとevaluator.pyのrag_comprehensive_metricを比較し、評価ロジックの違いを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - rag_optimization_gepa.pyを読んだか: 1点
   - evaluator.pyを読んだか: 1点

2. 必須要素の言及（6点）
   - gepa_metric_with_feedback関数の特定: 1点
   - rag_comprehensive_metric関数の特定: 1点
   - gepa_metric_with_feedbackの戻り値（dspy.Prediction with score and feedback）: 1.5点
   - rag_comprehensive_metricの戻り値（float score）: 1点
   - gepa_metric_with_feedbackの評価基準（レポート品質、構造、エラーメッセージ）: 1点
   - rag_comprehensive_metricの評価基準（exact_match、semantic_similarity、context_relevance）: 0.5点

3. 情報統合（2点）
   - 両メトリクスの目的の違い（GEPA最適化用 vs RAG性能評価用）を説明: 1点
   - フィードバック生成の有無と、それがGEPAリフレクションに与える影響を説明: 1点

オプション（加点）:
   - README.mdで評価メトリクスの背景を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Test Task 3: Hard (5 files) =====
        dspy.Example(
            task="JQaRAデータセットの完全なデータフローを追跡せよ。dataset_loader.pyでの読み込み → rag_module.pyでの処理 → evaluator.pyでの評価 → rag_optimization_gepa.pyでの最適化 → rag_evaluation.pyでの比較評価 の流れを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - dataset_loader.pyを読んだか: 0.5点
   - rag_module.pyを読んだか: 0.5点
   - evaluator.pyを読んだか: 0.5点
   - rag_optimization_gepa.pyを読んだか: 1点
   - rag_evaluation.pyを読んだか: 0.5点

2. 必須要素の言及（5点）
   - load_jqara_dataset関数によるJSONデータ読み込み: 0.5点
   - dspy.Example形式への変換（question、answer、context、is_positive）: 0.5点
   - positive/negativeデータの分離: 0.5点
   - RAGQAクラスによる処理（RewriteQuery → GenerateAnswer）: 1点
   - rag_comprehensive_metricによる評価: 0.5点
   - dspy.GEPA最適化（optimizer.compile）: 1点
   - rag_evaluation.pyでのベースライン vs 最適化モデル比較: 1点

3. 情報統合（2点）
   - データフロー全体（JSON → Example → RAG処理 → 評価 → 最適化 → 比較）を説明: 1点
   - trainset/valsetの分割と、それぞれの役割（学習用 vs 評価用）を説明: 1点

オプション（加点）:
   - README.mdでJQaRAデータセット詳細とGEPA最適化結果を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== Test Task 4: Medium (3 files) =====
        dspy.Example(
            task="rag_optimization_gepa.pyのモデル保存メカニズムを調査せよ。optimized_agent.save、タイムスタンプ付きファイル名生成、symlink作成（artifact/agent_gepa_optimized_latest.json）の仕組みを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - rag_optimization_gepa.pyを読んだか: 2点

2. 必須要素の言及（6点）
   - optimized_agent.save(model_path)の呼び出し: 1点
   - タイムスタンプ生成（datetime.now().strftime("%Y%m%d_%H%M")）: 1点
   - スコア付きファイル名生成（agent_gepa_optimized_YYYYMMDD_HHMM_scoreXXX.json）: 1.5点
   - artifact/ディレクトリの作成（os.makedirs）: 0.5点
   - symlink作成（os.symlink）: 1点
   - 既存symlinkの削除（os.remove）: 1点

3. 情報統合（2点）
   - モデル保存の目的（バージョン管理、最新版への簡単アクセス）を説明: 1点
   - タイムスタンプとスコアを含めることの利点を説明: 1点

オプション（加点）:
   - artifact/ディレクトリで実際の保存ファイルを確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Test Task 5: Hard (10 files) =====
        dspy.Example(
            task="27ディレクトリのGEPA最適化サイクル全体を包括的に調査せよ。全8個のPythonファイル、README.md、CLAUDE.mdを読み、データ読み込み → RAGパイプライン実装 → 評価メトリクス定義 → GEPA最適化実行 → モデル保存 → 評価比較 の全フローを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（5点）
   - config.pyを読んだか: 0.5点
   - dataset_loader.pyを読んだか: 0.5点
   - embeddings_cache.pyを読んだか: 0.5点
   - evaluator.pyを読んだか: 0.5点
   - rag_evaluation.pyを読んだか: 0.5点
   - rag_module.pyを読んだか: 0.5点
   - rag_optimization.pyを読んだか: 0.5点
   - rag_optimization_gepa.pyを読んだか: 0.5点
   - README.mdを読んだか: 0.5点
   - CLAUDE.mdを読んだか: 0.5点

2. 必須要素の言及（3点）
   - データ読み込み（load_jqara_dataset → dspy.Example変換）: 0.5点
   - RAGパイプライン（RewriteQuery → GenerateAnswer → RAGQA）: 0.5点
   - 評価メトリクス（exact_match_metric、rag_comprehensive_metric、gepa_metric_with_feedback）: 0.5点
   - GEPA最適化（dspy.GEPA、reflection_lm、Pareto戦略）: 0.5点
   - モデル保存（タイムスタンプ、スコア、symlink）: 0.5点
   - 評価比較（rag_evaluation.py、ベースライン vs 最適化）: 0.5点

3. 情報統合（2点）
   - 最適化サイクル全体（データ → RAG → 評価 → 最適化 → 保存 → 比較）を説明: 1点
   - 2モデル戦略（SMART_MODEL + FAST_MODEL）とPareto最適化の意義を説明: 1点

オプション（加点）:
   - README.mdでGEPA最適化結果（20%改善、35x効率化）の具体的数値を確認: +0.5点
   - pyproject.tomlで依存関係の全体像を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),
    ]

    return examples


def load_mini_test_dataset() -> List[dspy.Example]:
    """
    Load mini test dataset for quick GEPA optimization verification.

    3 representative examples from training dataset:
    - Easy: 1 task (Training Task 1)
    - Medium: 1 task (Training Task 3)
    - Hard: 1 task (Training Task 6)

    Returns:
        List of dspy.Example instances with task, working_directory, and criteria fields
    """
    # Get full training dataset
    full_train = load_training_dataset()

    # Select representative examples: Task 1 (Easy), Task 3 (Medium), Task 6 (Hard)
    mini_examples = [
        full_train[0],  # Task 1: Easy (SMART_MODEL/FAST_MODEL tracking)
        full_train[2],  # Task 3: Medium (JQaRA dataset conversion)
        full_train[5],  # Task 6: Hard (Full RAG system overview)
    ]

    return mini_examples


def load_file_exploration_dataset(
    dataset_type: str = "train",
    random_seed: int = 42
) -> List[dspy.Example]:
    """
    Load file exploration dataset (train, test, or mini_test).

    Args:
        dataset_type: "train" (10 examples), "test" (5 examples), or "mini_test" (3 examples)
        random_seed: Random seed for reproducibility (currently unused, for future expansion)

    Returns:
        List of dspy.Example instances
    """
    if dataset_type == "train":
        examples = load_training_dataset()
        print(f"📚 File Exploration Training Dataset loaded:")
        print(f"  Training examples: {len(examples)}")
        print(f"  Easy: 2, Medium: 5, Hard: 3")
    elif dataset_type == "test":
        examples = load_test_dataset()
        print(f"🧪 File Exploration Test Dataset loaded:")
        print(f"  Test examples: {len(examples)}")
        print(f"  Easy: 1, Medium: 2, Hard: 2")
    elif dataset_type == "mini_test":
        examples = load_mini_test_dataset()
        print(f"⚡ File Exploration Mini Test Dataset loaded:")
        print(f"  Mini test examples: {len(examples)}")
        print(f"  Easy: 1, Medium: 1, Hard: 1")
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Use 'train', 'test', or 'mini_test'.")

    return examples


def create_custom_example(
    task: str,
    working_directory: str = ".",
    criteria: str = "",
    difficulty: str = "medium"
) -> dspy.Example:
    """
    Create a custom file exploration example.

    Args:
        task: Task description
        working_directory: Working directory path
        criteria: Evaluation criteria (detailed scoring rubric)
        difficulty: Task difficulty ("easy", "medium", "hard")

    Returns:
        dspy.Example instance
    """
    return dspy.Example(
        task=task,
        working_directory=working_directory,
        criteria=criteria,
        difficulty=difficulty
    ).with_inputs("task", "working_directory")
