"""調査・情報収集に特化したエージェント"""

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


@tool
def web_search(query: str) -> str:
    """ウェブ検索を実行します（モック実装）"""
    return f"「{query}」に関する検索結果: 複数の調査レポートによると、2025年のグローバルAI市場規模は約5,200億ドルと予測されています（IDC調査）。2025年の最新AI技術として、生成AI（GPT、Claude）、マルチモーダルAI、エッジAI、量子機械学習などが注目されています。"


@tool
def get_wikipedia_info(topic: str) -> str:
    """Wikipedia情報を取得します（モック実装）"""
    return f"「{topic}」に関するWikipedia情報: {topic}は興味深いトピックです。"


def create_research_agent() -> CompiledGraph:
    """調査エージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")  # type: ignore[call-arg]

    tools = [web_search, get_wikipedia_info]

    prompt = """あなたは情報収集専門のエージェントです。

役割：
- 検索ツールを使って情報収集のみ行う
- 収集した情報（数値を含む）を端的に報告する

制約：
- 計算や分析は一切行わない
- 検索結果をそのまま報告する"""

    agent = create_react_agent(
        model=model, tools=tools, name="research_expert", prompt=prompt
    )

    return agent
