from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.memory import memory  # シングルトンインスタンスをインポート
from ..utils.todo_tools import (
    create_get_my_todos_for_agent,
    update_todo_status,
    update_multiple_todo_status,
)  # TODOツールをインポート
from ..utils.search_tools import (
    batch_compressed_search,
    get_search_results,
)  # 検索ツールをインポート


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
    return f"{topic}の調査結果を圧縮して保存しました"


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


def create_research_agent(model: BaseChatModel):
    # researchエージェント専用のget_todos関数
    get_research_todos = create_get_my_todos_for_agent("research")

    return create_react_agent(
        model=model,
        tools=[
            batch_compressed_search,
            get_search_results,
            get_research_todos,
            update_multiple_todo_status,
        ],
        name="research",
        prompt=(
            f"現在日付: {datetime.now().strftime('%Y年%m月%d日')}\n"
            "\n"
            "調査エージェント。効率的に調査を実行。\n"
            "\n"
            "手順:\n"
            "1. get_research_todos\n"
            "2. batch_compressed_search（全TODO並列検索・自動保存）\n"
            "3. get_search_results（結果確認・十分性判断）\n"
            "4. 判断結果:\n"
            "   - 十分: update_multiple_todo_status（全COMPLETED）→「調査完了」\n"
            "   - 不十分: batch_compressed_search（追加検索）→ステップ3に戻る\n"
            "\n"
            "重要: 情報が不足している場合は追加調査を実行"
        ),
    )


# グラフのエクスポート（LangGraph Studio用）
_model = ChatAnthropic(model_name="claude-sonnet-4-20250514")  # type: ignore
graph = create_research_agent(_model)
