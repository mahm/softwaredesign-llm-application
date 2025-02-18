from typing import List

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from ..llm import get_llm
from ..states.schemas import EvaluationResult, SearchQuery
from ..utils import load_prompt


class EvaluationOutput(BaseModel):
    """評価出力のスキーマ"""

    score: float
    improvement_points: List[str]
    required_searches: List[SearchQuery]


def run(content: str) -> EvaluationResult:
    """
    記事を評価する

    Args:
        content: 評価対象の記事内容
    Returns:
        EvaluationResult: 評価結果
    """
    llm = get_llm()
    prompt = load_prompt("evaluator")
    chain = prompt | llm | JsonOutputParser(pydantic_object=EvaluationOutput)

    result = chain.invoke({"article": content})

    return EvaluationResult(
        score=result.score,
        improvement_points=result.improvement_points,
        required_searches=result.required_searches,
    )
