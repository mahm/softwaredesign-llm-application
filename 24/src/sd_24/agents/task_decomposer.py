from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.todo_tools import (
    create_todo_task,
    create_multiple_todos,
)
from ..utils.search_tools import (
    search_and_save,
    get_search_results,
)


def create_task_decomposer():
    # モデルの定義
    model = ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=0,
    )

    # ツールの定義
    tools = [
        search_and_save,
        get_search_results,
        create_todo_task,
        create_multiple_todos,
    ]

    # システムプロンプトの定義
    system_prompt = f"""現在日付: {datetime.now().strftime('%Y年%m月%d日')}

あなたはタスク分解専門のアシスタントです。

役割:
- ユーザーの要求を分析し、実行可能なタスクに分解
- テーマに関する情報を収集し、適切な計画を立案
- 調査タスクはagent='research'、執筆タスクはagent='writer'として割り当て

重要な制約:
- タスク分解とTODO作成に専念
- まとめや総括は行わない
- TODOを作成したら「TODOを作成しました」とだけ報告"""

    return create_react_agent(
        name="task_decomposer",
        model=model,
        tools=tools,
        prompt=system_prompt,
    )


# グラフのエクスポート（LangGraph Studio用）
graph = create_task_decomposer()
