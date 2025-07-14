from typing import Annotated
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from datetime import datetime
import os
from pydantic import BaseModel, Field
from ..utils.memory import memory
from ..utils.todo_tools import (
    create_get_my_todos_for_agent,
    update_multiple_todo_status,
    TodoStatusUpdate,
)


class ContentEvaluation(BaseModel):
    """コンテンツ評価の構造化出力"""
    is_comprehensive: bool = Field(description="最終成果物として保存すべき包括的なレポートかどうか")
    quality_assessment: str = Field(description="コンテンツの品質評価")
    save_recommendation: str = Field(description="保存方法の推奨（file/memory）")


@tool
def get_writer_tasks() -> dict:
    """ライティングタスクの一覧を取得する

    このツールは現在割り当てられているライティングタスクを確認するために使用します。
    タスクの状態（pending/completed）や詳細情報を取得できます。

    Returns:
        dict: タスク情報を含む辞書
            - tasks: 未完了タスクのリスト
            - count: 未完了タスク数
            - has_tasks: タスクが存在するかどうか
            - next_task: 次に実行すべきタスク（存在する場合）
    """
    get_writer_todos = create_get_my_todos_for_agent("writer")
    todos_result = get_writer_todos.invoke({})
    todo_tasks = todos_result.get("tasks", [])
    pending_tasks = [
        task for task in todo_tasks if task.get("status") == "pending"]

    return {
        "tasks": pending_tasks,
        "count": len(pending_tasks),
        "has_tasks": len(pending_tasks) > 0,
        "next_task": pending_tasks[0] if pending_tasks else None
    }


class DataSufficiencyAssessment(BaseModel):
    """データ充足性評価の構造化出力"""
    sufficient: bool = Field(description="執筆に十分なデータがあるかどうか")
    reasoning: str = Field(description="判断の理由")
    missing_aspects: str = Field(description="不足している観点（もしあれば）")
    recommendation: str = Field(description="推奨アクション")


@tool
async def check_research_data_sufficiency() -> dict:
    """研究データが執筆に十分かどうかを評価する

    このツールは執筆を開始する前に、利用可能な研究データが十分かどうかを判断します。
    データの量、質、カバレッジをLLMが総合的に評価し、執筆可能かどうかを判定します。

    Returns:
        dict: 評価結果を含む辞書
            - sufficient: 執筆に十分なデータがあるか（bool）
            - topics: 利用可能なトピックのリスト
            - topic_count: トピック数
            - total_content_length: 総文字数
            - reasoning: 判断の理由
            - missing_aspects: 不足している観点
            - recommendation: 推奨アクション
    """
    raw_research_data = memory.get("research", {})
    research_data = {}

    for topic, data in raw_research_data.items():
        if isinstance(data, dict) and "findings" in data:
            research_data[topic] = data["findings"]
        elif isinstance(data, str):
            research_data[topic] = data
        else:
            research_data[topic] = str(data)

    # データの概要情報
    topics = list(research_data.keys())
    total_content_length = sum(len(str(content))
                               for content in research_data.values())

    # LLMに評価させる
    data_summary = "\n".join(
        [f"## {topic}\n{content[:500]}..." for topic, content in research_data.items()])

    # 現在のタスクを取得して文脈を理解
    get_writer_todos = create_get_my_todos_for_agent("writer")
    todos_result = get_writer_todos.invoke({})
    todo_tasks = todos_result.get("tasks", [])
    pending_tasks = [
        task for task in todo_tasks if task.get("status") == "pending"]

    task_context = ""
    if pending_tasks:
        task_descriptions = [task.get("description", "")
                             for task in pending_tasks[:3]]
        task_context = "\n\n予定されているタスク:\n" + \
            "\n".join([f"- {desc}" for desc in task_descriptions])

    prompt = f"""以下の研究データが、執筆タスクを実行するために十分かどうか評価してください。{task_context}

現在のデータ:
- トピック数: {len(topics)}
- 総文字数: {total_content_length}
- トピック一覧: {', '.join(topics)}

データの内容（抜粋）:
{data_summary}

以下を判断してください：
1. sufficient: 執筆に十分なデータがあるか（true/false）
2. reasoning: その判断の具体的な理由
3. missing_aspects: もし不足があれば、何が足りないか
4. recommendation: 次のアクション（"執筆可能" or "追加調査が必要：〇〇について"）
"""

    model = ChatAnthropic(  # type: ignore
        model_name="claude-sonnet-4-20250514",
        temperature=0.3
    )

    try:
        structured_model = model.with_structured_output(
            DataSufficiencyAssessment)
        assessment = await structured_model.ainvoke([HumanMessage(content=prompt)])

        assert isinstance(assessment, DataSufficiencyAssessment)
        return {
            "sufficient": assessment.sufficient,
            "topics": topics,
            "topic_count": len(topics),
            "total_content_length": total_content_length,
            "reasoning": assessment.reasoning,
            "missing_aspects": assessment.missing_aspects,
            "recommendation": assessment.recommendation
        }
    except Exception:
        # フォールバック
        return {
            "sufficient": False,
            "topics": topics,
            "topic_count": len(topics),
            "total_content_length": total_content_length,
            "reasoning": "評価エラー",
            "recommendation": "システムエラーのため再試行が必要"
        }


@tool
async def write_and_save_content(
    task_id: Annotated[str, "タスクID"],
    task_description: Annotated[str, "執筆するタスクの説明"]
) -> dict:
    """タスクに基づいてコンテンツを執筆し、適切な場所に保存する

    このツールは指定されたタスクの内容に基づいて、研究データを活用して
    高品質な記事やレポートを生成します。生成したコンテンツは、
    その性質に応じて最終成果物としてファイルに保存するか、
    中間成果物としてメモリに保存します。

    Args:
        task_id: 実行するタスクの識別子
        task_description: タスクの詳細な説明（何を執筆するか）

    Returns:
        dict: 実行結果を含む辞書
            - task_id: 処理したタスクID
            - content_preview: 生成したコンテンツの冒頭部分
            - word_count: 文字数
            - is_comprehensive: 包括的なレポートかどうか
            - saved_to: 保存先（"file" または "memory"）
            - filepath: ファイル保存の場合のパス（該当する場合）
            - memory_key: メモリ保存の場合のキー（該当する場合）
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
        temperature=0.7
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])
    content = response.content
    assert isinstance(content, str)

    # LLMにコンテンツを評価させる
    eval_prompt = f"""以下のタスクとコンテンツを評価してください。

タスク: {task_description}
コンテンツの文字数: {len(content)}

コンテンツの冒頭部分:
{content[:1000]}...

以下を判断してください：
1. is_comprehensive: これは最終成果物として保存すべき包括的なレポートか（true/false）
2. quality_assessment: コンテンツの品質についての評価
3. save_recommendation: 保存方法の推奨（"file"=最終成果物としてファイル保存、"memory"=中間成果物としてメモリ保存）
"""

    try:
        eval_model = model.with_structured_output(ContentEvaluation)
        evaluation = await eval_model.ainvoke([HumanMessage(content=eval_prompt)])
        assert isinstance(evaluation, ContentEvaluation)
        is_comprehensive = evaluation.is_comprehensive
        save_recommendation = evaluation.save_recommendation
    except Exception:
        # フォールバック: タスク説明から推測
        is_comprehensive = len(content) >= 2000
        save_recommendation = "file" if is_comprehensive else "memory"

    # 保存処理
    if save_recommendation == "file":
        # 最終成果物としてファイルに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", timestamp)
        os.makedirs(output_dir, exist_ok=True)

        filename = "article.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()

        # メモリにも保存
        memory.set("final_document", content)
        memory.set("final_document_path", filepath)

        save_info = {"saved_to": "file", "filepath": filepath}
    else:
        # 中間成果物としてメモリに保存
        memory_key = f"section_{task_id}"
        memory.set(memory_key, content)
        save_info = {"saved_to": "memory", "memory_key": memory_key}

    return {
        "task_id": task_id,
        "content_preview": content[:200] + "...",
        "word_count": len(content),
        "is_comprehensive": is_comprehensive,
        **save_info,
        "success": True
    }


@tool
async def complete_writing_task(task_id: Annotated[str, "タスクID"]) -> dict:
    """執筆タスクを完了状態に更新する

    このツールは指定されたタスクを完了状態にマークします。
    タスクの執筆と保存が正常に終了した後に使用してください。

    Args:
        task_id: 完了するタスクの識別子

    Returns:
        dict: 更新結果を含む辞書
            - task_id: 処理したタスクID
            - status: 更新後のステータス（"completed"）
            - description: タスクの説明
            - success: 更新成功フラグ
            - error: エラーメッセージ（失敗時のみ）
    """
    try:
        # タスクの詳細を取得
        get_writer_todos = create_get_my_todos_for_agent("writer")
        todos_result = get_writer_todos.invoke({})
        todo_tasks = todos_result.get("tasks", [])

        task = next((t for t in todo_tasks if t["id"] == task_id), None)
        if not task:
            return {"success": False, "error": "タスクが見つかりません"}

        # 完了状態に更新
        await update_multiple_todo_status.ainvoke({
            "updates": [
                TodoStatusUpdate(
                    task_id=task_id,
                    completed=True,
                    result=f"執筆完了: {task['description'][:50]}..."
                )
            ]
        })

        return {
            "task_id": task_id,
            "status": "completed",
            "description": task["description"],
            "success": True
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "error": str(e),
            "success": False
        }


@tool
def check_all_tasks_completed() -> dict:
    """すべてのライティングタスクが完了したかを確認する

    このツールは全体の進捗状況を確認し、すべてのタスクが完了したかどうかを
    チェックします。また、最終成果物がファイルに保存されている場合は、
    そのパスも返します。

    Returns:
        dict: 完了状況を含む辞書
            - all_completed: すべてのタスクが完了したか（bool）
            - total_tasks: 総タスク数
            - completed_count: 完了したタスク数
            - pending_count: 未完了のタスク数
            - final_document_path: 最終成果物のファイルパス（存在する場合）
            - status: 現在の状況を説明するメッセージ
    """
    get_writer_todos = create_get_my_todos_for_agent("writer")
    todos_result = get_writer_todos.invoke({})
    todo_tasks = todos_result.get("tasks", [])

    pending_tasks = [
        task for task in todo_tasks if task.get("status") == "pending"]
    completed_tasks = [
        task for task in todo_tasks if task.get("status") == "completed"]

    all_completed = len(pending_tasks) == 0 and len(todo_tasks) > 0

    # 最終成果物のパスを取得
    final_path = memory.get("final_document_path", None)

    return {
        "all_completed": all_completed,
        "total_tasks": len(todo_tasks),
        "completed_count": len(completed_tasks),
        "pending_count": len(pending_tasks),
        "final_document_path": final_path,
        "status": "全タスク完了" if all_completed else f"{len(pending_tasks)}件のタスクが残っています"
    }


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

あなたは熟練したライティングエージェントです。与えられたタスクに基づいて高品質なコンテンツを執筆します。

## あなたの役割
- 研究データを分析・統合して、読みやすく有益な記事やレポートを作成する
- タスクの要求に応じて適切な形式と長さのコンテンツを生成する
- 生成したコンテンツを適切な場所（ファイルまたはメモリ）に保存する

## 基本的な作業の流れ
1. まず研究データの充足性を確認する
2. タスクを1つずつ取得して執筆する
3. 各タスクの完了後、そのタスクを完了状態にする
4. すべてのタスクが終わったら最終確認を行う

## 重要事項
- 研究データが不足している場合は、その旨を報告する
- 最終成果物のファイルパスは必ず報告する
- エラーが発生した場合は詳細を含めて報告する"""

    agent = create_react_agent(
        name="writer",
        model=model,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


# グラフのエクスポート
graph = create_writer_agent()
