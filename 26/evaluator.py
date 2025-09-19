"""共通評価モジュール"""
import dspy # type: ignore

from config import configure_lm, configure_embedder, FAST_MODEL
from embeddings_cache import get_cached_embeddings_retriever


def rag_comprehensive_metric(gold, pred, trace=None):
    """RAGシステムの総合評価メトリクス

    回答の完全一致（70%）と検索の再現率（30%）を組み合わせた評価
    """
    # 回答の完全一致
    answer_match = float(pred.answer.strip() == gold.answer.strip())

    # 検索精度の評価（正例の再現率）
    if hasattr(pred, 'retrieved_passages') and hasattr(gold, 'positives'):
        retrieved = set(pred.retrieved_passages) if pred.retrieved_passages else set()
        positives = set(gold.positives) if gold.positives else set()

        # 再現率: 正例のうち何割を検索できたか
        recall = len(retrieved & positives) / len(positives) if positives else 1.0

        # 総合スコア: 回答70% + 検索30%
        return 0.7 * answer_match + 0.3 * recall

    # retrieved_passagesまたはpositivesがない場合は回答のみで評価
    return answer_match


def evaluation(rag_module, examples, corpus_texts, num_questions=30, display_table=5):
    """testセットで評価を実行"""
    # 設定
    dspy.configure(
        lm=configure_lm(FAST_MODEL, temperature=0.0, max_tokens=1000),
        rm=get_cached_embeddings_retriever(
            embedder=configure_embedder(),
            corpus_texts=corpus_texts,
            k=10
        )
    )

    # 評価実行
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=rag_comprehensive_metric,
        num_threads=4,
        display_progress=True,
        display_table=display_table
    )

    return evaluator(rag_module)