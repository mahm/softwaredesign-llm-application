"""検索ツール with 圧縮機能"""

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Optional
import asyncio
from pydantic import BaseModel, Field
from .memory import memory


class SearchQuery(BaseModel):
    """検索クエリ用のデータモデル"""

    query: str = Field(..., description="検索クエリ")
    topic: Optional[str] = Field(
        default=None, description="保存時のキー（省略時はqueryを使用）"
    )


@tool  # No return_direct - part of sequential flow
async def batch_compressed_search(queries: list[SearchQuery]) -> str:
    """Web検索を並列実行し、結果を圧縮して外部メモリに保存

    Args:
        queries: SearchQueryのリスト。単一検索の場合は1要素のリストを渡す

    Returns:
        各検索の完了メッセージ
    """

    # 各検索タスクを非同期で作成
    async def search_and_compress(search_query: SearchQuery) -> Optional[str]:
        query = search_query.query
        topic = search_query.topic if search_query.topic else query

        if not query:
            return None

        # Tavily検索を実行（非同期）
        search = TavilySearch(max_results=5)
        search_response = await search.ainvoke(query)

        # 検索結果を文字列に整形
        formatted_results = f"検索クエリ: {query}\n\n"

        if isinstance(search_response, dict) and "results" in search_response:
            results = search_response["results"]
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    formatted_results += f"結果 {i}:\n"
                    formatted_results += f"タイトル: {result.get('title', 'N/A')}\n"
                    formatted_results += f"URL: {result.get('url', 'N/A')}\n"
                    formatted_results += f"内容: {result.get('content', 'N/A')}\n\n"
        else:
            formatted_results += f"検索結果: {str(search_response)}\n\n"

        # 結果を圧縮して保存
        compressed_results = await memory.compress_research(topic, formatted_results)
        search_results = memory.get("search_results", {})
        search_results[topic] = compressed_results
        memory.set("search_results", search_results)

        return f"「{topic}」: 検索完了"

    # すべての検索を並列実行
    tasks = [search_and_compress(query) for query in queries]
    results = await asyncio.gather(*tasks)

    # 結果をフィルタリング（Noneを除外）
    results_summary = [r for r in results if r is not None]

    if results_summary:
        return (
            f"{len(results_summary)}件の検索が並列実行で完了しました:\n"
            + "\n".join(results_summary)
            + "\n\nget_search_resultsで結果を取得できます。"
        )
    else:
        return "検索クエリが指定されていません。"


@tool  # No return_direct - part of sequential flow
async def get_search_results(topics: Optional[str | list[str]] = None) -> str:
    """保存された圧縮検索結果を取得（単一/複数対応）

    Args:
        topics: 取得したい検索結果のキー。文字列または文字列のリスト。省略時はトピック一覧を返す

    Returns:
        圧縮された検索結果または利用可能なトピック一覧
    """
    search_results = memory.get("search_results", {})

    if not search_results:
        return "保存された検索結果はありません。compressed_searchまたはbatch_compressed_searchを使用して検索を実行してください。"

    if topics:
        # 単一のトピックの場合はリストに変換
        if isinstance(topics, str):
            topics = [topics]

        results = []
        not_found = []

        for topic in topics:
            result = search_results.get(topic)
            if result:
                results.append(f"## 「{topic}」の圧縮検索結果:\n{result}")
            else:
                not_found.append(topic)

        output = ""
        if results:
            output = "\n\n".join(results)

        if not_found:
            available_topics = list(search_results.keys())
            if output:
                output += "\n\n"
            output += f"以下のトピックは見つかりませんでした: {', '.join(not_found)}\n"
            output += f"利用可能なトピック: {', '.join(available_topics)}"

        return output if output else "指定されたトピックの検索結果はありません。"
    else:
        # トピックが指定されていない場合は一覧表示
        topics_list = "\n".join([f"- {key}" for key in search_results.keys()])
        return f"保存されている検索結果のトピック:\n{topics_list}\n\n特定のトピックを指定してget_search_resultsを呼び出してください。"
