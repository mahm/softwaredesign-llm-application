from ..current_date import current_date
from ..llm import get_structured_llm
from ..states.schemas import EvaluationResult
from ..utils import load_prompt


def run(content: str, threshold: float = 70) -> EvaluationResult:
    """
    記事の評価を行う

    Args:
        content: 評価対象の記事内容
    Returns:
        EvaluationResult: 評価結果
    """
    chain = load_prompt("evaluator") | get_structured_llm(output_type=EvaluationResult)
    return chain.invoke(
        {
            "current_date": current_date,
            "content": content,
            "threshold": threshold,
        }
    )
