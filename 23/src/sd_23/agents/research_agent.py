"""調査・情報収集に特化したエージェント"""

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


@tool
def web_search(query: str) -> str:
    """ウェブ検索を実行します（モック実装）"""
    return f"「{query}」に関する検索結果: 複数の関連情報が見つかりました。"


@tool
def get_wikipedia_info(topic: str) -> str:
    """Wikipedia情報を取得します（モック実装）"""
    return f"「{topic}」に関するWikipedia情報: {topic}は興味深いトピックです。"


def create_research_agent() -> CompiledGraph:
    """調査エージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")  # type: ignore[call-arg]

    tools = [web_search, get_wikipedia_info]

    prompt = """あなたは調査・情報収集の専門家です。
与えられたトピックについて、利用可能なツールを使用して情報を収集し、
わかりやすく要約して提供してください。"""

    agent = create_react_agent(
        model=model, tools=tools, name="research_expert", prompt=prompt
    )

    return agent
