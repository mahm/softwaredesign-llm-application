"""技術サポートに特化したエージェント"""

import random
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool
from langgraph.graph.graph import CompiledGraph


@tool
def check_system_status() -> str:
    """システムステータスをチェックします（モック実装）"""
    return "システムステータス: すべてのサービスが正常に動作しています。"


@tool
def create_support_ticket(issue_description: str) -> str:
    """サポートチケットを作成します（モック実装）"""
    ticket_id = f"TICKET-{random.randint(1000, 9999)}"
    return f"サポートチケット {ticket_id} を作成しました。技術チームが確認後、ご連絡いたします。"


def create_tech_agent() -> CompiledGraph:
    """技術サポートエージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")  # type: ignore[call-arg]

    # FAQサポートへのハンドオフツール
    faq_handoff = create_handoff_tool(
        agent_name="faq_support",
        description="一般的な質問やFAQに関する内容はFAQサポートに転送",
    )

    tools = [check_system_status, create_support_ticket, faq_handoff]

    prompt = """あなたは技術サポート担当です。
技術的な問題の診断と解決をサポートします。

システムステータスの確認や、複雑な問題に対してはサポートチケットを
作成することができます。

一般的な質問やFAQで解決できそうな内容の場合は、
faq_supportに転送してください。

専門的でありながら、わかりやすい説明を心がけてください。"""

    agent = create_react_agent(
        model=model, tools=tools, name="tech_support", prompt=prompt
    )

    return agent
