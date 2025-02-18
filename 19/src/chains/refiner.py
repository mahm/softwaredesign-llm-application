from langchain_core.output_parsers import StrOutputParser

from ..llm import get_llm
from ..states.schemas import RefinerResult
from ..utils import load_prompt


def run(content: str, style: str) -> RefinerResult:
    """
    記事を指定されたスタイルで改善する

    Args:
        content: 改善対象の記事内容
        style: 改善のスタイル（"保守的"、"中間"、"積極的"）
    Returns:
        RefinerResult: diff形式の改善内容
    """
    llm = get_llm()
    prompt = load_prompt("refiner")
    chain = prompt | llm | StrOutputParser()

    diff = chain.invoke({"draft": content, "style": style})

    return RefinerResult(diff=diff, style=style)
