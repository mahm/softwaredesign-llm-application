"""共通評価モジュール"""
import dspy # type: ignore

from config import configure_lm, configure_embedder, FAST_MODEL


def exact_match(gold, pred, trace=None):
    """完全一致メトリクス"""
    return float(pred.answer.strip() == gold.answer.strip())


def evaluation(rag_module, examples, corpus_texts, num_questions=30, display_table=5):
    """testセットで評価を実行"""
    # 設定
    dspy.configure(
        lm=configure_lm(FAST_MODEL, temperature=0.0, max_tokens=1000),
        rm=dspy.retrievers.Embeddings(
            embedder=configure_embedder(),
            corpus=corpus_texts,
            k=10
        )
    )

    # 評価実行
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=exact_match,
        num_threads=4,
        display_progress=True,
        display_table=display_table
    )

    return evaluator(rag_module)