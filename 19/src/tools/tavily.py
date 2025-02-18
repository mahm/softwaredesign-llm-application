import os
from functools import lru_cache

from tavily import TavilyClient


@lru_cache(maxsize=1)
def client() -> TavilyClient:
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 10,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    """Tavilyを使用してウェブ検索を実行します。

    Args:
        query: 検索クエリ
        search_depth: 検索の深さ（"basic"または"advanced"）
        topic: 検索カテゴリ（"general"または"news"）
        max_results: 返す結果の最大数（5-20）
        include_domains: 検索対象とするドメインのリスト
        exclude_domains: 検索から除外するドメインのリスト
    """
    try:
        response = client().search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
        )

        # 結果のフォーマット
        output = []
        if response.get("answer"):
            output.append(f"Answer: {response['answer']}\n")
            output.append("Sources:")
            for result in response["results"]:
                output.append(f"- {result['title']}: {result['url']}")
            output.append("")

        output.append("Detailed Results:")
        for result in response["results"]:
            output.append(f"\nTitle: {result['title']}")
            output.append(f"URL: {result['url']}")
            output.append(f"Content: {result['content']}")

        return "\n".join(output)
    except Exception as e:
        return f"検索に失敗しました: {str(e)}"
