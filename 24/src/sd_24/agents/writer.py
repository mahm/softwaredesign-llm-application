from typing import Annotated
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from datetime import datetime
import os
from ..utils.memory import memory
from ..utils.todo_tools import (
    create_get_my_todos_for_agent,
    update_multiple_todo_status,
    TodoStatusUpdate,
)


@tool(return_direct=True)
async def check_research_data_sufficiency() -> dict:
    """研究データが執筆に十分かどうかを簡易チェック"""
    research_data = memory.get("research", {})
    
    # シンプルな判定：トピックが1つ以上あり、内容があれば執筆可能
    topics = list(research_data.keys())
    sufficient = len(topics) > 0 and any(research_data.values())
    
    return {
        "sufficient": sufficient,
        "topic_count": len(topics),
        "recommendation": "執筆可能" if sufficient else "追加調査が必要"
    }


@tool(return_direct=True)
async def write_and_save_content(
    task_id: Annotated[str, "タスクID"],
    task_description: Annotated[str, "執筆するタスクの説明"]
) -> dict:
    """タスクに基づいてコンテンツを執筆し、ファイルに保存する

    このツールは指定されたタスクの内容に基づいて、研究データを活用して
    高品質な記事やレポートを生成し、タスクID付きのMarkdownファイルとして保存します。

    Args:
        task_id: 実行するタスクの識別子
        task_description: タスクの詳細な説明（何を執筆するか）

    Returns:
        dict: 実行結果を含む辞書
            - task_id: 処理したタスクID
            - content_preview: 生成したコンテンツの冒頭部分
            - word_count: 文字数
            - filepath: 保存されたファイルのパス
            - success: 実行成功フラグ
    """
    # 研究データを取得
    raw_research_data = memory.get("research", {})
    research_data = {}

    for topic, data in raw_research_data.items():
        if isinstance(data, dict) and "findings" in data:
            research_data[topic] = data["findings"]
        elif isinstance(data, str):
            research_data[topic] = data
        else:
            research_data[topic] = str(data)

    # プロンプト構築
    research_sections = "\n".join(
        [f"## {topic}\n{content}\n" for topic, content in research_data.items()])

    prompt = f"""あなたは熟練した執筆者です。以下のタスクについて高品質な記事を執筆してください。

タスク: {task_description}

利用可能な調査データ:
{research_sections}

要件:
- 調査データを分析・統合して有機的な記事として構成
- 総合的なレポートの場合は2000文字以上
- タイトル、序文、本文、結論を含む構成
- 日本の情報源についても言及
"""

    model = ChatAnthropic(  # type: ignore
        model_name="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=8192
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])
    content = response.content
    assert isinstance(content, str)

    # ファイルに保存（タスクID付きファイル名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{task_id}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()

    # メモリには最終成果物のパスのみ保存
    memory.set("final_document_path", filepath)
    
    # 作成されたファイルパスをリストに追加
    created_files = memory.get("created_files", [])
    created_files.append(filepath)
    memory.set("created_files", created_files)

    return {
        "task_id": task_id,
        "content_preview": content[:200] + "...",
        "word_count": len(content),
        "filepath": filepath,
        "success": True
    }


@tool(return_direct=True)
async def complete_writing_task(task_id: Annotated[str, "タスクID"]) -> str:
    """執筆タスクを完了状態に更新"""
    await update_multiple_todo_status.ainvoke({
        "updates": [TodoStatusUpdate(task_id=task_id, completed=True, result="執筆完了")]
    })
    return f"タスク {task_id} を完了しました"


@tool(return_direct=True)
async def check_all_tasks_completed() -> str:
    """すべてのライティングタスクが完了したかを確認"""
    get_writer_todos = create_get_my_todos_for_agent("writer")
    todos_result = get_writer_todos.invoke({})
    todo_tasks = todos_result.get("tasks", [])
    
    pending_count = len([t for t in todo_tasks if t.get("status") == "pending"])
    final_path = memory.get("final_document_path", "")
    
    if pending_count == 0:
        return f"全タスク完了！ファイル: {final_path}"
    return f"残り{pending_count}件のタスクがあります"


def create_writer_agent():
    # モデルの定義
    model = ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=0,
    )

    # ツールの定義
    tools = [
        create_get_my_todos_for_agent("writer"),
        check_research_data_sufficiency,
        write_and_save_content,
        complete_writing_task,
        check_all_tasks_completed,
    ]

    # システムプロンプトの定義
    system_prompt = f"""現在日付: {datetime.now().strftime('%Y年%m月%d日')}

あなたは執筆エージェントです。研究データを使って記事を作成し、ファイルに保存します。

作業手順:
1. 研究データの確認
2. タスクを取得して執筆
3. タスクを完了に更新
4. 全タスク完了を確認"""

    agent = create_react_agent(
        name="writer",
        model=model,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


# グラフのエクスポート
graph = create_writer_agent()
