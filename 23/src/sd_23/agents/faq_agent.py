"""FAQ対応に特化したエージェント"""

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool
from langgraph.graph.graph import CompiledGraph


@tool
def check_faq_database(question: str) -> str:
    """FAQ データベースをチェックします（モック実装）"""
    faq_responses = {
        "password": "パスワードのリセットは、ログイン画面の「パスワードを忘れた」リンクから行えます。",
        "billing": "請求に関しては、アカウント設定の「請求情報」セクションをご確認ください。",
        "account": "アカウントの作成は、トップページの「新規登録」ボタンから行えます。",
    }

    for keyword, response in faq_responses.items():
        if keyword in question.lower():
            return response

    return "申し訳ございません。FAQデータベースに該当する情報が見つかりませんでした。"


def create_faq_agent() -> CompiledGraph:
    """FAQエージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")  # type: ignore[call-arg]

    # 技術サポートへのハンドオフツール
    tech_handoff = create_handoff_tool(
        agent_name="tech_support",
        description="技術的な問題や詳細なサポートが必要な場合に技術サポートに転送",
    )

    tools = [check_faq_database, tech_handoff]

    prompt = """あなたはFAQサポート担当です。
よくある質問に対して、FAQデータベースを検索して回答してください。

FAQで解決できない技術的な問題や、より詳細なサポートが必要な場合は、
tech_supportに転送してください。

常に丁寧で親切な対応を心がけてください。"""

    agent = create_react_agent(
        model=model, tools=tools, name="faq_support", prompt=prompt
    )

    return agent
