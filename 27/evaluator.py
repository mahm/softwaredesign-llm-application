"""共通評価モジュール"""
import dspy # type: ignore

from config import configure_lm, configure_embedder, FAST_MODEL, RETRIEVAL_K
from embeddings_cache import get_cached_embeddings_retriever


def exact_match_metric(gold, pred, trace=None):
    """メトリクス関数（評価用）"""
    return float(pred.answer.strip() == gold.answer.strip())


def rag_comprehensive_metric(gold, pred, trace=None):
    """メトリクス関数（最適化用）"""
    # 回答の完全一致を評価
    answer_match = float(pred.answer.strip() == gold.answer.strip())

    # 正答が含まれる割合に応じて評価
    retrieved = set(pred.retrieved_passages)
    positives = set(gold.positives) if gold.positives else set()
    max_positives = min(len(positives), RETRIEVAL_K) if positives else 0
    positive_ratio = len(retrieved & positives) / max_positives if max_positives else 0.0

    # 総合スコア: 回答50% + 検索50%
    return 0.5 * answer_match + 0.5 * positive_ratio


def evaluation(rag_module, examples, corpus_texts, display_table=5):
    """testセットで評価を実行"""
    # 設定
    dspy.configure(
        lm=configure_lm(FAST_MODEL, temperature=0.0, max_tokens=4096),
        rm=get_cached_embeddings_retriever(
            embedder=configure_embedder(),
            corpus_texts=corpus_texts,
            k=RETRIEVAL_K
        )
    )

    # 評価実行（完全一致のみを評価）
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=exact_match_metric,
        num_threads=4,
        display_progress=True,
        display_table=display_table
    )

    return evaluator(rag_module)