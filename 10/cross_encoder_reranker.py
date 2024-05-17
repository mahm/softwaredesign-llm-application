import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
            self,
            model_name: str = "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
            default_activation_function=None,
            use_fp16: bool = True
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            model_name,
            device=device,
            default_activation_function=default_activation_function,
        )
        if use_fp16 and "cuda" in device:
            self.model.model.half()
        self.model.max_length = 512

    def rerank(self, query: str, documents: list[str]) -> tuple[list[int], list[float]]:
        # クエリと各ドキュメントをペアにする
        pairs = []
        for doc in documents:
            pair = [query, doc]
            pairs.append(pair)

        # 各ペアに対してモデルを使用してスコアを予測する
        predicted_scores = self.model.predict(pairs)

        # 予測されたスコアを浮動小数点数に変換する
        scores = []
        for score in predicted_scores:
            scores.append(float(score))

        # スコアに基づいてドキュメントのインデックスを降順にソートする
        indices = list(range(len(scores)))  # インデックスのリストを作成
        indices.sort(key=lambda i: scores[i], reverse=True)  # スコアに基づいてインデックスをソート

        # ソートされたインデックスとスコアを返す
        return indices, scores
