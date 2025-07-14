from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.todo_tools import (
    create_get_my_todos_for_agent,
    update_multiple_todo_status,
)
from ..utils.search_tools import (
    search_and_save,
    get_search_results,
)


def create_research_agent():
    # モデルの定義
    model = ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=0,
    )

    # ツールの定義
    tools = [
        create_get_my_todos_for_agent("research"),
        search_and_save,
        get_search_results,
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
