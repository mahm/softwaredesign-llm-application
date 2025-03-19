import json
import os

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient  # type: ignore

import src.mcp_servers.database as db

# MCPサーバーの初期化
mcp = FastMCP("knowledge-db-mcp-server")

# --- ツール定義 ---

# 1. Tavily APIを使ったWeb検索ツール
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in environment")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Tavily APIを使用してWeb検索を行い、上位の結果を返します。

    引数:
        query: 検索クエリ
        max_results: 返す検索結果の最大数 (デフォルト: 5)

    返値:
        検索結果のテキスト（各結果のタイトル・URL・スニペット）
    """
    try:
        response = tavily_client.search(
            query, search_depth="moderate", max_results=max_results
        )
    except Exception as e:
        return f"検索エラー: {e}"

    answer = response.get("answer")
    result_text = ""
    if answer:
        result_text += f"回答: {answer}\n\n"

    results = response.get("results", [])
    if not results:
        return result_text + "検索結果が見つかりませんでした。"

    for i, res in enumerate(results, start=1):
        title = res.get("title", "(タイトルなし)")
        url = res.get("url", "")
        snippet = res.get("content") or res.get("snippet") or ""
        result_text += f"{i}. {title}\n   URL: {url}\n"
        if snippet:
            result_text += f"   概要: {snippet}\n\n"

    return result_text.strip()


@mcp.tool()
def save_search_result(
    query: str,
    url: str,
    title: str,
    content: str,
    content_type: str = "",
    summary: str = "",
    tags: str = "",
    reliability_score: float = 0.5,
) -> str:
    """
    検索結果をデータベースに保存します。

    引数:
        query: 検索クエリ
        url: 情報ソースのURL
        title: コンテンツのタイトル
        content: 抽出したコンテンツ（本文）
        content_type: 情報タイプ（例: "ニュース", "技術文書"）
        summary: 要約文（エージェントが生成）
        tags: カンマ区切りのタグ
        reliability_score: 信頼性スコア (0.0-1.0)

    返値:
        保存処理の結果メッセージ
    """
    result = db.save_search_result(
        query, url, title, content, content_type, summary, tags, reliability_score
    )
    return result["message"]


@mcp.tool()
def get_recent_results(days: int = 7, limit: int = 10, content_type: str = "") -> str:
    """
    指定された日数以内の最近の検索結果を取得します。

    引数:
        days: 何日前までの検索結果を取得するか (デフォルト: 7)
        limit: 返す結果の最大数 (デフォルト: 10)
        content_type: 特定のコンテンツタイプでフィルタリング (オプション)

    返値:
        最近の検索結果とそのサマリー（JSON形式）
    """
    result = db.get_recent_results(days, limit, content_type)
    if not result["success"]:
        return result["message"]

    return json.dumps(result["results"], ensure_ascii=False, indent=2)


@mcp.tool()
def get_content_by_id(result_id: int) -> str:
    """
    特定IDの検索結果の詳細コンテンツを取得します。

    引数:
        result_id: 検索結果のID

    返値:
        検索結果の詳細（タイトル、URL、コンテンツなど）
    """
    result = db.get_content_by_id(result_id)
    if not result["success"]:
        return result["message"]

    return json.dumps(result["result"], ensure_ascii=False, indent=2)


@mcp.tool()
def get_content_types() -> str:
    """
    データベースに保存されている全てのコンテンツタイプの一覧と各タイプの件数を返します。

    返値:
        コンテンツタイプとその件数の一覧（JSON形式）
    """
    result = db.get_content_types()
    if not result["success"]:
        return result["message"]

    return json.dumps(result["results"], ensure_ascii=False, indent=2)


@mcp.tool()
def get_schema() -> str:
    """
    SQLiteデータベースのスキーマ情報（テーブル名と各カラム）を返します。
    この情報は、LLMが適切なクエリを作成するためのヒントとして利用されます。
    """
    result = db.get_schema()
    if not result["success"]:
        return result["message"]

    return result["schema"]


@mcp.tool()
def select_query(query: str) -> str:
    """
    SQLiteデータベースに対してSELECTクエリを実行し、結果を返します。
    例: "SELECT * FROM search_results WHERE content_type='ニュース' LIMIT 10;"
    ※SELECT文のみ許可されています。
    """
    result = db.execute_select_query(query)
    if not result["success"]:
        return result["message"]

    return json.dumps(result["results"], ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("MCPサーバーを起動します...")
    mcp.run(transport="stdio")
