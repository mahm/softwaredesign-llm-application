"""
DeepEval搭載メトリクスの組み立て。judgeは全メトリクスでgpt-5.4に統一する。
G-Evalはスコアをトークンのlogprobsで加重するため、judgeにはlogprobs対応モデルが必要。
"""

from deepeval.metrics import GEval, SummarizationMetric, ToolCorrectnessMetric
from deepeval.test_case import SingleTurnParams

JUDGE_MODEL = "gpt-5.4"


# 網羅性の判定質問。既定では実行のたびに自動生成されるためブレの原因になる。
# 「原文でyesと答えられる質問」を固定し、要約が同じ答えを保っているかを測る
ASSESSMENT_QUESTIONS = [
    "この文章は、論文が解決しようとする課題や既存手法の限界を述べているか?",
    "この文章は、提案手法の中核的な仕組みを説明しているか?",
    "この文章は、主要な実験結果を具体的な数値とともに示しているか?",
    "この文章は、実験の規模・データセット・計算資源など再現に関わる情報を含んでいるか?",
    "この文章は、論文の貢献または結論を述べているか?",
]


def build_metrics() -> list:
    # 取得: 期待したツール(論文取得のexecuteとgenerate_pptx)が呼ばれたか
    tool_correctness = ToolCorrectnessMetric()

    # 要約: score = min(整合性, 網羅性)。input=論文本文, actual_output=スライドテキスト
    summarization = SummarizationMetric(
        threshold=0.5,
        model=JUDGE_MODEL,
        assessment_questions=ASSESSMENT_QUESTIONS,
    )

    # GEvalによるスライド品質評価
    slide_quality = GEval(
        name="SlideQuality",
        evaluation_steps=[
            "actual_outputは論文から生成されたプレゼンスライドである。各スライドが1つのメッセージに絞られているかを確認し、1枚に複数の論点が詰め込まれていれば減点する",
            "箇条書きの密度を確認する。1スライドに5項目以上ある場合や、1項目が60字を超えて長い場合は「文字の壁」として減点する",
            "スライドタイトルを確認する。「提案手法」「実験結果」のような章名だけのタイトルは減点し、そのスライドの主張を含む具体的なタイトルは加点する",
            "全体が背景と課題→提案手法→結果→結論の論理的な流れで並んでいるかを確認する",
            "省略自体は要約として許容する。内容の欠落よりも、詰め込み・冗長・曖昧なタイトルを重く扱う",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=JUDGE_MODEL,
    )

    return [tool_correctness, summarization, slide_quality]
