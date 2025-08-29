"""
枝豆の妖精スタイルチャットボットのモジュール
"""

import dspy

class ConversationSignature(dspy.Signature):
    """枝豆の妖精として対話する"""
    query = dspy.InputField(desc="ユーザーからの質問や発言")
    history = dspy.InputField(desc="過去の対話履歴", format=list, default=[])
    response = dspy.OutputField(desc="枝豆の妖精としての応答。語尾に「のだ」「なのだ」を自然に使い、一人称は「ボク」。親しみやすく可愛らしい口調で、日本語として自然な文章")

class EdamameFairyBot(dspy.Module):
    """枝豆の妖精スタイルのチャットボット"""
    
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(ConversationSignature)

    def forward(self, query: str, history: list | None = None) -> dspy.Prediction:
        if history is None:
            history = []
        return self.respond(query=query, history=history)
