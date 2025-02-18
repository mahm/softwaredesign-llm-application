from langchain_core.output_parsers import StrOutputParser

from ..llm import get_llm
from ..utils import load_prompt


def run(query: str, search_result: str) -> str:
    """
    質問に対する記事を生成する

    Args:
        query: ユーザーからの質問
    Returns:
        str: 生成された記事
    """
    chain = load_prompt("writer") | get_llm() | StrOutputParser()
    return chain.invoke({"query": query, "search_result": search_result})
