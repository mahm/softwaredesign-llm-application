"""評価スクリプト"""
from rag_module import RAGQA
from rag_optimization import OPTIMIZED_MODEL_LATEST
from evaluator import evaluation
from dataset_loader import load_jqara_dataset


def main():
    testset, test_corpus_texts = load_jqara_dataset(num_questions=30, dataset_split='test')

    # ベースライン評価
    baseline = RAGQA()
    base_results = evaluation(baseline, examples=testset, corpus_texts=test_corpus_texts, display_table=0)

    # 最適化モデル評価
    optimized = RAGQA()
    optimized.load(OPTIMIZED_MODEL_LATEST)
    opt_results = evaluation(optimized, examples=testset, corpus_texts=test_corpus_texts, display_table=0)
    print("=" * 60)
    print("🔬 RAG評価")
    print(f"[Baseline]  EM: {base_results.score:.1f}%")
    print(f"[Optimized] EM: {opt_results.score:.1f}% (Δ {opt_results.score - base_results.score:+.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
