from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.memory import memory
from ..utils.todo_tools import (
    create_get_my_todos_for_agent,
    update_multiple_todo_status,
)
from ..utils.search_tools import (
    batch_compressed_search,
    get_search_results,
)


@tool  # No return_direct - part of sequential flow
async def save_research(topic: str, findings: str, needs_revision: bool = False) -> str:
    """調査結果を圧縮して保存し、必要に応じて再調査を提案"""
    # Web検索結果を圧縮
    compressed = await memory.compress_research(topic, findings)
    # 既存の調査結果に追加
    research_data = memory.get("research", {})
    research_data[topic] = {
        "findings": compressed,
        "needs_revision": needs_revision,
        "timestamp": str(datetime.now()),
    }
    memory.set("research", research_data)

    if needs_revision:
        return f"{topic}の調査結果を保存しました。ただし、情報が不十分なため再調査を推奨します。"
    return f"{topic}の調査結果を保存しました"


@tool  # No return_direct - let agent report the result
def check_research_sufficiency() -> dict:
    """調査結果の充足度をチェックし、追加調査の必要性を判断"""
    research_data = memory.get("research", {})

    insufficient_topics = []
    for topic, data in research_data.items():
        if isinstance(data, dict) and data.get("needs_revision"):
            insufficient_topics.append(topic)

    return {
        "sufficient": len(insufficient_topics) == 0,
        "insufficient_topics": insufficient_topics,
        "recommendation": "すべての調査が完了しています"
        if not insufficient_topics
        else f"以下のトピックについて追加調査が必要です: {', '.join(insufficient_topics)}",
    }


def create_research_agent():
    # モデルの定義
    model = ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=0,
    )

    # ツールの定義
    tools = [
        create_get_my_todos_for_agent("research"),
        batch_compressed_search,
        get_search_results,
        save_research,
        update_multiple_todo_status,
    ]

    # プロンプトの定義
    system_prompt = f"""現在日付: {datetime.now().strftime('%Y年%m月%d日')}

あなたは情報収集専門のアシスタントです。

役割:
- task_decomposerが作成した計画に従って調査を実行
- 情報収集とメモリへの保存に専念
- 情報が不足している場合は追加調査を実行

重要な制約:
- レポート執筆や文章作成は行わない（Writerの役割）
- 調査結果のまとめや要約は出力しない
- 検索結果を保存したら「調査完了」とだけ報告
- 計画が不適切な場合は、その旨を簡潔に報告"""

    return create_react_agent(
        name="research",
        model=model,
        tools=tools,
        prompt=system_prompt,
    )


# グラフのエクスポート（LangGraph Studio用）
graph = create_research_agent()
