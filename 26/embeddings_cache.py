"""
Embeddingsのキャッシュ機能
コーパスのEmbedding計算結果をキャッシュして再利用
"""

import pickle
import hashlib
from pathlib import Path
import dspy  # type: ignore


def get_cached_embeddings_retriever(
    embedder,
    corpus_texts,
    k=10,
    cache_dir="artifact/embeddings_cache"
):
    """キャッシュ機能付きEmbeddings Retrieverを取得

    同一コーパスのEmbeddingは一度だけ計算し、以降はキャッシュから読み込む

    Args:
        embedder: DSPy Embedderインスタンス
        corpus_texts: 検索対象のテキストコーパス
        k: 検索結果数（デフォルト10）
        cache_dir: キャッシュディレクトリ（デフォルト: artifact/embeddings_cache）

    Returns:
        dspy.retrievers.Embeddings: キャッシュから復元または新規作成したRetriever
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # コーパスのハッシュ値でキャッシュファイルを識別
    # ソートして順序に依存しないハッシュ値を生成
    corpus_content = "".join(sorted(corpus_texts))
    corpus_hash = hashlib.md5(corpus_content.encode()).hexdigest()[:12]
    cache_file = cache_path / f"embeddings_{corpus_hash}.pkl"

    if cache_file.exists():
        # キャッシュから読み込み
        print(f"📂 キャッシュからEmbeddingを読み込み: {cache_file.name}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # Retrieverを作成（Embedding計算をスキップ）
            retriever = dspy.retrievers.Embeddings(
                embedder=embedder,
                corpus=corpus_texts,
                k=k
            )

            # キャッシュされたベクトルを直接設定
            # DSPyのEmbeddingsはcorpus_embeddings属性にベクトルを保存
            if hasattr(retriever, 'corpus_embeddings'):
                retriever.corpus_embeddings = cached_data['vectors']
            else:
                # フォールバック: 属性名が変更された場合
                print("⚠️ キャッシュされたベクトルの設定に失敗。再計算します。")
                return _create_and_cache_retriever(embedder, corpus_texts, k, cache_file)

            print(f"  ✅ キャッシュから{len(corpus_texts)}件のEmbeddingを復元")

        except Exception as e:
            print(f"⚠️ キャッシュの読み込みに失敗: {e}")
            # キャッシュが破損している場合は再作成
            return _create_and_cache_retriever(embedder, corpus_texts, k, cache_file)

    else:
        # 新規作成してキャッシュ
        retriever = _create_and_cache_retriever(embedder, corpus_texts, k, cache_file)

    return retriever


def _create_and_cache_retriever(embedder, corpus_texts, k, cache_file):
    """Retrieverを新規作成してキャッシュに保存

    Args:
        embedder: DSPy Embedderインスタンス
        corpus_texts: 検索対象のテキストコーパス
        k: 検索結果数
        cache_file: キャッシュファイルのパス

    Returns:
        dspy.retrievers.Embeddings: 新規作成したRetriever
    """
    print(f"🔄 {len(corpus_texts)}件のEmbeddingを計算中...")

    # Retrieverを作成（Embedding計算が実行される）
    retriever = dspy.retrievers.Embeddings(
        embedder=embedder,
        corpus=corpus_texts,
        k=k
    )

    # ベクトルを取得してキャッシュに保存
    vectors_to_cache = None
    if hasattr(retriever, 'corpus_embeddings'):
        vectors_to_cache = retriever.corpus_embeddings

    if vectors_to_cache is not None:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'vectors': vectors_to_cache,
                    'corpus_size': len(corpus_texts)
                }, f)
            print(f"💾 Embeddingをキャッシュに保存: {cache_file.name}")
        except Exception as e:
            print(f"⚠️ キャッシュの保存に失敗: {e}")
            # 保存に失敗してもRetrieverは返す
    else:
        print("⚠️ ベクトルの取得に失敗。キャッシュは作成されません。")

    return retriever


def clear_embeddings_cache(cache_dir="artifact/embeddings_cache"):
    """キャッシュをクリア

    Args:
        cache_dir: キャッシュディレクトリ
    """
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for cache_file in cache_path.glob("embeddings_*.pkl"):
            cache_file.unlink()
            print(f"🗑️ キャッシュを削除: {cache_file.name}")
        print("✅ すべてのEmbeddingキャッシュをクリアしました")
    else:
        print("ℹ️ キャッシュディレクトリが存在しません")