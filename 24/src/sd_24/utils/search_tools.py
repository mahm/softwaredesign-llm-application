"""シンプルな検索ツール"""

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from .memory import memory


@tool
async def search_and_save(query: str, topic: str) -> str:
    """Web検索を実行し、結果を保存"""    
    search = TavilySearch(max_results=3)
    response = await search.ainvoke(query)
    
    # 検索結果を整形
    content = f"検索: {query}\n\n"
    if isinstance(response, dict) and "results" in response:
        for i, result in enumerate(response["results"], 1):
            if isinstance(result, dict):
                content += f"{i}. {result.get('title', '')}\n"
                content += f"{result.get('content', '')}\n\n"
    
    # 結果を圧縮して保存
    compressed = await memory.compress_research(topic, content)
    search_results = memory.get("search_results", {})
    search_results[topic] = compressed
    memory.set("search_results", search_results)
    
    return f"「{topic}」の検索完了"


@tool
def get_search_results(topic: str | None = None) -> str:
    """保存された検索結果を取得"""
    search_results = memory.get("search_results", {})
    
    if not search_results:
        return "検索結果がありません"
        
    if topic:
        result = search_results.get(topic)
        return result if result else f"「{topic}」の結果がありません"
    else:
        topics = list(search_results.keys())
        return f"保存されているトピック: {', '.join(topics)}"
