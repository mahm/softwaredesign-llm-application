from langchain_core.output_parsers import StrOutputParser

from ..llm import get_llm
from ..states.schemas import WriterResult
from ..utils import load_prompt


def run(query: str, search_result: str) -> WriterResult:
    """
    検索結果から記事を生成する

    Args:
        query: ユーザーからの質問
        search_result: 検索結果のテキスト
    Returns:
        WriterResult: 生成された記事の情報
    """
    chain = load_prompt("writer") | get_llm() | StrOutputParser()

    content = chain.invoke({"query": query, "search_result": search_result})

    return WriterResult(
        content=content,
        sources=[],  # TODO: 情報源の抽出
        structure=[],  # TODO: 記事構造の抽出
    )
