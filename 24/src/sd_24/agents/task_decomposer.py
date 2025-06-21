from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.memory import memory  # シングルトンインスタンスをインポート
from ..utils.todo_tools import (
    create_todo_task,
    create_multiple_todos,
    get_todo_progress,
)  # TODOツールをインポート
from ..utils.search_tools import (
    batch_compressed_search,
    get_search_results,
)  # 検索ツールをインポート


@tool  # No return_direct - agent needs to evaluate feedback
def get_research_feedback() -> dict:
    """調査結果のフィードバックを取得して計画修正の必要性を判断"""
    research_data = memory.get("research", {})
    feedback = {
        "has_feedback": bool(research_data),
        "topics_with_issues": [],
        "suggestions": [],
    }

    for topic, data in research_data.items():
        if isinstance(data, dict) and data.get("needs_revision"):
            feedback["topics_with_issues"].append(topic)  # type: ignore
            feedback["suggestions"].append(f"{topic}についてより具体的な調査項目を追加")  # type: ignore

    return feedback


def create_task_decomposer(model: BaseChatModel):
    return create_react_agent(
        model=model,
        tools=[
            batch_compressed_search,
            create_todo_task,
            create_multiple_todos,
        ],
        name="task_decomposer",
        prompt=(
            f"現在日付: {datetime.now().strftime('%Y年%m月%d日')}\n"
            "\n"
            "タスク分解エージェント。\n"
            "\n"
            "手順:\n"
            "1. batch_compressed_search（テーマ構成情報収集）\n"
            "2. create_todo_task（親タスク）\n"
            "3. create_multiple_todos:\n"
            "   - 調査: 最新動向、具体例、課題\n"
            "   - 執筆: 序論、本論、結論\n"
            "4. 「TODOを作成しました」報告\n"
            "\n"
            "最適化: get_research_feedback, get_todo_progress省略"
        ),
    )


# グラフのエクスポート（LangGraph Studio用）
_model = ChatAnthropic(model_name="claude-sonnet-4-20250514")  # type: ignore
graph = create_task_decomposer(_model)
