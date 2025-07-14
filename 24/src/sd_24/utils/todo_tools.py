from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from .todo_manager import todo_manager, TaskStatus
from typing import Optional
from pydantic import BaseModel, Field


class TodoTaskInput(BaseModel):
    """TODO作成用の入力データモデル"""

    description: str = Field(..., description="タスクの説明")
    agent: str = Field(..., description="担当エージェント (research または writer)")
    parent_task_id: Optional[str] = Field(
        default=None, description="親タスクID（オプション）"
    )


class TodoStatusUpdate(BaseModel):
    """TODOステータス更新用の入力データモデル"""

    task_id: str = Field(..., description="タスクID")
    completed: bool = Field(..., description="完了フラグ")
    result: Optional[str] = Field(default="", description="実行結果（オプション）")


def create_get_my_todos_for_agent(agent_name: str) -> Runnable:
    """特定のエージェント用のget_my_todos関数を作成"""

    @tool  # No return_direct - part of sequential flow
    def get_agent_todos() -> dict:
        """このエージェントの未完了TODOを取得"""
        pending_tasks = todo_manager.get_pending_tasks(agent_name)
        return {
            "tasks": [task.to_dict() for task in pending_tasks],
            "count": len(pending_tasks),
        }

    # 関数名とドキュメントを動的に設定
    get_agent_todos.name = f"get_{agent_name}_todos"  # type: ignore
    # type: ignore
    get_agent_todos.description = f"{agent_name}エージェントの未完了TODOを取得"
    return get_agent_todos




@tool  # No return_direct - part of sequential flow
def update_todo_status(update: TodoStatusUpdate) -> str:
    """TODOタスクのステータスを更新"""
    status = TaskStatus.COMPLETED if update.completed else TaskStatus.IN_PROGRESS
    todo_manager.update_status(update.task_id, status, update.result)
    return f"TODO {update.task_id} を{status.value}に更新しました"




@tool  # No return_direct - part of sequential flow
def create_todo_task(task: TodoTaskInput) -> str:
    """TODOタスクを作成"""
    task_id = todo_manager.add_task(
        description=task.description, agent=task.agent, parent_id=task.parent_task_id
    )
    return f"TODOタスク作成: {task_id} - {task.description}"


@tool
async def create_multiple_todos(tasks: list[TodoTaskInput]) -> str:
    """複数のTODOタスクを一度に作成"""
    created_tasks = []
    for task in tasks:
        task_id = todo_manager.add_task(
            description=task.description,
            agent=task.agent,
            parent_id=task.parent_task_id,
        )
        created_tasks.append(f"{task_id}: {task.description} ({task.agent})")
    
    return f"{len(created_tasks)}件のTODOタスクを作成しました:\n" + "\n".join(created_tasks)


@tool
async def update_multiple_todo_status(updates: list[TodoStatusUpdate]) -> str:
    """複数のTODOタスクのステータスを一度に更新"""
    for update in updates:
        status = TaskStatus.COMPLETED if update.completed else TaskStatus.IN_PROGRESS
        todo_manager.update_status(update.task_id, status, update.result)
    return f"{len(updates)}件のタスクを更新しました"
