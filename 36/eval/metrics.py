"""DeepEval搭載メトリクスの組み立て。judgeは全メトリクスでgpt-5.4に統一する。

G-Evalはスコアをトークンのlogprobsで加重するため、judgeにはlogprobs対応モデルが必要。
"""

from deepeval.metrics import GEval, SummarizationMetric, ToolCorrectnessMetric
from deepeval.test_case import SingleTurnParams

JUDGE_MODEL = "gpt-5.4"


def build_metrics() -> list:
    # 取得: 期待したツール(論文取得のexecuteとgenerate_pptx)が呼ばれたか
    tool_correctness = ToolCorrectnessMetric()

    # 要約中核: score = min(整合性, 網羅性)。input=論文本文, actual_output=スライドテキスト
    summarization = SummarizationMetric(threshold=0.5, model=JUDGE_MODEL)

    # 見せ方: 評価基準を自然文で記述するだけでメトリクスになる
    slide_quality = GEval(
        name="SlideQuality",
        criteria=(
            "actual_output は論文から生成されたプレゼンスライドである。"
            "各スライドが1つの論点に絞られているか、全体が動機->手法->結果->結論の論理的な流れを持つか、"
            "タイトルが具体的で情報量があるかを評価する。省略は要約として許容し、過剰な文字量は減点する。"
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=JUDGE_MODEL,
    )

    return [tool_correctness, summarization, slide_quality]
