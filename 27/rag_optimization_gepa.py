"""
RAG最適化スクリプト (GEPA版)
"""

import os
import sys
import random
import argparse
import logging
import dspy # type: ignore
from datetime import datetime

from config import configure_lm, configure_embedder, SMART_MODEL, FAST_MODEL, RETRIEVAL_K
from rag_module import RAGQA
from dataset_loader import load_jqara_dataset
from evaluator import evaluation, rag_comprehensive_metric
from embeddings_cache import get_cached_embeddings_retriever

# 最適化されたモデルの保存先（最新版へのリンク）
GEPA_OPTIMIZED_MODEL_LATEST = "artifact/rag_gepa_optimized_latest.json"


class Tee:
    """標準出力をコンソールとファイルの両方に出力するクラス"""

    def __init__(self, file_path: str, original_stdout):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = original_stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def gepa_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA用のメトリクス関数（スコア + テキストフィードバック）

    Args:
        gold: 正解データ
        pred: 予測結果
        trace: プログラム全体の実行トレース（オプション）
        pred_name: 最適化中の特定のpredictorの名前（オプション）
        pred_trace: 特定のpredictorの実行トレース（オプション）

    Returns:
        dspy.Prediction: ScoreWithFeedback型（scoreとfeedbackフィールドを持つ）
    """
    # 基本スコア計算
    score = rag_comprehensive_metric(gold, pred, trace)

    # フィードバック生成
    feedback_parts = []

    # 回答の評価
    if pred.answer.strip() == gold.answer.strip():
        feedback_parts.append("✓ 完全一致")
    else:
        # 部分一致チェック
        pred_lower = pred.answer.strip().lower()
        gold_lower = gold.answer.strip().lower()
        if gold_lower in pred_lower or pred_lower in gold_lower:
            feedback_parts.append(f"△ 部分一致: 期待={gold.answer}, 実際={pred.answer}")
        else:
            feedback_parts.append(f"✗ 不正解: 期待={gold.answer}, 実際={pred.answer}")

    # 検索精度のフィードバック
    retrieved = set(pred.retrieved_passages) if hasattr(pred, 'retrieved_passages') else set()
    positives = set(gold.positives) if hasattr(gold, 'positives') and gold.positives else set()

    if positives:
        overlap = len(retrieved & positives)
        max_retrievable = min(len(positives), RETRIEVAL_K)
        recall = overlap / max_retrievable if max_retrievable > 0 else 0

        if recall >= 0.8:
            feedback_parts.append(f"✓ 検索良好: {overlap}/{max_retrievable}個の正解文書")
        elif recall >= 0.5:
            feedback_parts.append(f"△ 検索改善余地: {overlap}/{max_retrievable}個の正解文書")
        else:
            feedback_parts.append(f"✗ 検索不良: {overlap}/{max_retrievable}個の正解文書")

        # クエリ改善の提案
        if recall < 0.5 and hasattr(pred, 'rewritten_query'):
            feedback_parts.append(f"クエリ改善を検討: '{pred.rewritten_query}'")

    # predictor固有のフィードバック（もし指定されていれば）
    if pred_name:
        feedback_parts.append(f"[{pred_name}]")

    # フィードバック文字列の生成
    feedback = " | ".join(feedback_parts)

    # GEPAはScoreWithFeedback型（dspy.Prediction）を期待
    return dspy.Prediction(
        score=score,
        feedback=feedback
    )


def log_metric_evaluation(gold, pred, trace, pred_name, pred_trace, result):
    """メトリクス評価のログ記録

    Args:
        gold: 正解データ
        pred: 予測結果
        trace: プログラム全体の実行トレース
        pred_name: predictor名
        pred_trace: predictor実行トレース
        result: メトリクス計算結果（dspy.Prediction）
    """
    logger = logging.getLogger("gepa_optimization")

    # 入力引数の記録
    logger.info(f"gold = {repr(gold)}")
    logger.info(f"pred = {repr(pred)}")
    logger.info(f"trace = {repr(trace) if trace is not None else 'None'}")
    logger.info(f"pred_name = {repr(pred_name) if pred_name is not None else 'None'}")
    logger.info(f"pred_trace = {repr(pred_trace) if pred_trace is not None else 'None'}")

    logger.info("-" * 80)

    # 計算結果の記録
    logger.info(f"score = {result.score}")
    logger.info(f"feedback = {result.feedback}")
    logger.info("=" * 80)


def gepa_metric_with_feedback_logged(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """ロギング機能付きGEPAメトリクス関数

    Args:
        gold: 正解データ
        pred: 予測結果
        trace: プログラム全体の実行トレース（オプション）
        pred_name: 最適化中の特定のpredictorの名前（オプション）
        pred_trace: 特定のpredictorの実行トレース（オプション）

    Returns:
        dspy.Prediction: ScoreWithFeedback型（scoreとfeedbackフィールドを持つ）
    """
    result = gepa_metric_with_feedback(gold, pred, trace, pred_name, pred_trace)
    log_metric_evaluation(gold, pred, trace, pred_name, pred_trace, result)

    return result


def setup_logging(timestamp: str) -> tuple:
    """ロギング環境のセットアップ

    Args:
        timestamp: タイムスタンプ文字列（YYYYMMDD_HHMM形式）

    Returns:
        tuple: (original_stdout, tee, log_path, stdout_path)
    """
    # ログディレクトリの作成
    os.makedirs("logs", exist_ok=True)

    # ログファイルのパス
    log_filename = f"gepa_optimization_{timestamp}.log"
    log_path = os.path.join("logs", log_filename)

    # 標準出力ファイルのパス
    stdout_filename = f"gepa_optimization_{timestamp}_stdout.log"
    stdout_path = os.path.join("logs", stdout_filename)

    # ロガーの設定
    logger = logging.getLogger("gepa_optimization")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 標準出力のリダイレクト
    original_stdout = sys.stdout
    tee = Tee(stdout_path, original_stdout)
    sys.stdout = tee

    print(f"📝 ログファイル: {log_path}")
    print(f"📄 標準出力ログ: {stdout_path}")

    return original_stdout, tee, log_path, stdout_path


def cleanup_logging(original_stdout, tee: Tee, log_path: str, stdout_path: str) -> None:
    """ロギング環境のクリーンアップ

    Args:
        original_stdout: 元の標準出力
        tee: Teeオブジェクト
        log_path: ログファイルパス
        stdout_path: 標準出力ログパス
    """
    # ロガーのクリーンアップ
    logger = logging.getLogger("gepa_optimization")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # 標準出力を元に戻す
    sys.stdout = original_stdout
    tee.close()

    # ファイルパスを表示
    print("\n✅ 最適化完了！")
    print(f"📝 詳細ログ: {log_path}")
    print(f"📄 標準出力: {stdout_path}")


def main(seed=42):
    """メイン実行関数

    Args:
        seed: ランダムシード（デフォルト: 42）
    """
    # タイムスタンプ生成（ログファイル名とモデルファイル名で共有）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ロギング環境のセットアップ
    original_stdout, tee, log_path, stdout_path = setup_logging(timestamp)

    try:
        # データセット読み込み（devセットを使用）
        examples, corpus_texts = load_jqara_dataset(num_questions=50, dataset_split='dev', random_seed=seed)

        # Train/Val分割（50:50）
        random.seed(seed)  # 引数のシードを使用
        random.shuffle(examples)
        split = int(len(examples) * 0.5)
        trainset = examples[:split]
        valset = examples[split:]
        print(f"✂️ データ分割 (dev): train={len(trainset)}, val={len(valset)}")

        # testセット読み込み（30問）
        testset, test_corpus_texts = load_jqara_dataset(num_questions=30, dataset_split='test', random_seed=seed)

        # LM設定
        print("\n🔧 モデル設定中...")
        # GEPAのリフレクション用LM（高温度設定）
        reflection_lm = configure_lm(SMART_MODEL, temperature=1.0, max_tokens=8192)
        # 推論用（高速モデル）
        fast_lm = configure_lm(FAST_MODEL, temperature=0.0, max_tokens=4096)

        # 埋め込みモデル設定
        embedder = configure_embedder()

        # Retrieverの構築（キャッシュ機能付き）
        print("🔍 検索システムを構築中...")
        retriever = get_cached_embeddings_retriever(
            embedder=embedder,
            corpus_texts=corpus_texts,
            k=RETRIEVAL_K  # 検索結果数
        )

        # DSPyで使用するデフォルトを設定
        dspy.configure(lm=fast_lm, rm=retriever)

        # ベースライン評価（testセットで評価）
        print("\n📊 ベースライン評価中...")
        baseline = RAGQA()
        base_results = evaluation(baseline, examples=testset, corpus_texts=test_corpus_texts, display_table=0)

        # GEPA最適化
        print("\n🚀 GEPA最適化を開始...")

        # 最適化対象のRAGモジュール
        rag = RAGQA()

        # GEPAの設定
        optimizer = dspy.GEPA(
            metric=gepa_metric_with_feedback_logged,  # ロギング機能付きメトリクス
            auto="medium",  # 最適化の強度
            num_threads=4,  # 並列スレッド数
            reflection_minibatch_size=3,  # リフレクション時のミニバッチサイズ
            reflection_lm=reflection_lm,  # リフレクション用のLM（強力なモデル推奨）
            candidate_selection_strategy="pareto",  # パレート最適化戦略
            track_stats=True,  # 統計情報の追跡
        )

        # 最適化実行（RAGの推論はfast_lm、リフレクションはreflection_lmを使用）
        optimized_rag = optimizer.compile(
            rag,
            trainset=trainset,
            valset=valset,
        )

        # 最適化後の評価（testセット）
        print("\n📊 最適化後の評価中...")
        opt_results = evaluation(optimized_rag, examples=testset, corpus_texts=test_corpus_texts, display_table=0)
        print(f"  [Baseline] EM: {base_results.score:.1f}%")
        print(f"  [GEPA Optimized] EM: {opt_results.score:.1f}%")
        print(f"  改善: {opt_results.score - base_results.score:+.1f}%")

        # スコアを含むファイル名生成（testセットのスコアを使用、タイムスタンプは既存のものを利用）
        em_score_str = f"em{int(opt_results.score):03d}"
        model_filename = f"rag_gepa_optimized_{timestamp}_{em_score_str}.json"
        model_path = os.path.join("artifact", model_filename)

        # モデルの保存
        os.makedirs("artifact", exist_ok=True)
        optimized_rag.save(model_path)
        print(f"\n💾 最適化済みモデルを保存: {model_path}")

        # 最新版へのシンボリックリンク作成
        if os.path.exists(GEPA_OPTIMIZED_MODEL_LATEST):
            os.remove(GEPA_OPTIMIZED_MODEL_LATEST)
        os.symlink(model_filename, GEPA_OPTIMIZED_MODEL_LATEST)

        print(f"  → 最新リンク: {GEPA_OPTIMIZED_MODEL_LATEST}")

    finally:
        # ロギング環境のクリーンアップ
        cleanup_logging(original_stdout, tee, log_path, stdout_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAGの最適化 (GEPA版)')
    parser.add_argument('--seed', type=int, default=42,
                       help='ランダムシード（デフォルト: 42）')
    args = parser.parse_args()

    print(f"🌱 シード値: {args.seed}")
    print("🧬 最適化手法: GEPA (Genetic-Pareto)")
    main(seed=args.seed)