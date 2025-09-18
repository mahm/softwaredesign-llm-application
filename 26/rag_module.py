"""
RAGモジュール
検索クエリ最適化型RAGパイプラインの実装
"""

import dspy # type: ignore


class RewriteQuery(dspy.Signature):
    """質問文を検索に適した形にリライト"""
    question = dspy.InputField(desc="ユーザーからの質問")
    rewritten_query = dspy.OutputField(desc="検索に最適化された質問文")


class GenerateAnswer(dspy.Signature):
    """コンテキストと質問から回答を生成"""
    context = dspy.InputField(desc="検索で取得した文書群")
    question = dspy.InputField(desc="ユーザーからの質問")
    answer = dspy.OutputField(desc="質問に対する簡潔な回答")


class RAGQA(dspy.Module):
    """RAGパイプライン（QueryRewriter → Retrieve → Answer）"""

    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(RewriteQuery)
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        # 1) クエリ最適化
        rewritten = self.rewrite(question=question).rewritten_query

        # 2) 検索
        result = dspy.settings.rm(rewritten)
        passages = result.passages if hasattr(result, 'passages') else []

        # 3) コンテキスト作成
        context = "\n".join(passages) if passages else ""

        # 4) 回答生成
        answer = self.generate(context=context, question=question).answer

        # 結果を返す
        return dspy.Prediction(
            answer=answer,
            retrieved_passages=passages,
            rewritten_query=rewritten
        )
