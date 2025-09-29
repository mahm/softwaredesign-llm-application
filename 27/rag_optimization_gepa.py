"""
RAG最適化スクリプト (GEPA版)
"""

import os
import random
import argparse
import dspy # type: ignore
from datetime import datetime

from config import configure_lm, configure_embedder, SMART_MODEL, FAST_MODEL, RETRIEVAL_K
from rag_module import RAGQA
from dataset_loader import load_jqara_dataset
from evaluator import evaluation, rag_comprehensive_metric
from embeddings_cache import get_cached_embeddings_retriever

# 最適化されたモデルの保存先（最新版へのリンク）
GEPA_OPTIMIZED_MODEL_LATEST = "artifact/rag_gepa_optimized_latest.json"


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
    # 基本スコア計算（既存のメトリクスを使用）
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

    # GEPAはScoreWithFeedback型（dspy.Prediction）を期待
    return dspy.Prediction(
        score=score,
        feedback=" | ".join(feedback_parts)
    )


def main(seed=42):
    """メイン実行関数

    Args:
        seed: ランダムシード（デフォルト: 42）
    """

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
    # 最適化用（高性能モデル）
    smart_lm = configure_lm(SMART_MODEL, temperature=0.0, max_tokens=4096)
    # 推論用（高速モデル）
    fast_lm = configure_lm(FAST_MODEL, temperature=0.0, max_tokens=4096)

    # GEPAのリフレクション用LM（高温度設定）
    reflection_lm = configure_lm(SMART_MODEL, temperature=1.0, max_tokens=8192)

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
    print("  ※GEPAはリフレクティブな進化的アプローチでプロンプトを最適化します")

    # 最適化対象のRAGモジュール
    rag = RAGQA()

    # GEPAの設定
    optimizer = dspy.GEPA(
        metric=gepa_metric_with_feedback,  # フィードバック付きメトリクス
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

    # タイムスタンプとスコアを含むファイル名生成（testセットのスコアを使用）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAGの最適化 (GEPA版)')
    parser.add_argument('--seed', type=int, default=42,
                       help='ランダムシード（デフォルト: 42）')
    args = parser.parse_args()

    print(f"🌱 シード値: {args.seed}")
    print("🧬 最適化手法: GEPA (Gradient-Estimation with Prompt Augmentation)")
    main(seed=args.seed)