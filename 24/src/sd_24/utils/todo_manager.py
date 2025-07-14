"""TODOタスク管理モジュール"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """タスクのステータス"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TodoItem(BaseModel):
    """TODOアイテム"""

    id: str = Field(..., description="タスクID")
    description: str = Field(..., description="タスクの説明")
    agent: str = Field(..., description="担当エージェント")
    parent_id: Optional[str] = Field(None, description="親タスクID")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="タスクのステータス"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新日時")
    result: Optional[Any] = Field(None, description="タスクの結果")

    class Config:
        use_enum_values = False
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> Dict[str, Any]:
        """Pydanticモデルを辞書に変換"""
        data = self.model_dump()
        # statusを文字列に変換
        data["status"] = self.status.value
        # datetimeをISOフォーマットに変換
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data


class TodoManager:
    """TODO管理クラス"""

    def __init__(self):
        self.todos: Dict[str, TodoItem] = {}
        self.task_counter = 0

    def add_task(
        self, description: str, agent: str, parent_id: Optional[str] = None
    ) -> str:
        """タスクを追加"""
        self.task_counter += 1
        task_id = f"TASK-{self.task_counter:04d}"
        todo = TodoItem(
            id=task_id, description=description, agent=agent, parent_id=parent_id, result=None
        )
        self.todos[task_id] = todo
        return task_id

    def update_status(self, task_id: str, status: TaskStatus, result: Any = None):
        """タスクのステータスを更新"""
        if task_id in self.todos:
            self.todos[task_id].status = status
            self.todos[task_id].updated_at = datetime.now()
            if result:
                self.todos[task_id].result = result

    def get_pending_tasks(self, agent: Optional[str] = None) -> List[TodoItem]:
        """未完了タスクを取得"""
        tasks = []
        for todo in self.todos.values():
            if todo.status == TaskStatus.PENDING:
                if agent is None or todo.agent == agent:
                    tasks.append(todo)
        return tasks

    def get_task_tree(self) -> str:
        """タスクツリーを文字列で返す"""

        def build_tree(parent_id: Optional[str], indent: int = 0) -> List[str]:
            lines = []
            for todo in self.todos.values():
                if todo.parent_id == parent_id:
                    status_icon = {
                        TaskStatus.PENDING: "⬜",
                        TaskStatus.IN_PROGRESS: "🔄",
                        TaskStatus.COMPLETED: "✅",
                        TaskStatus.FAILED: "❌",
                    }[todo.status]

                    line = (
                        "  " * indent
                        + f"{status_icon} [{todo.id}] {todo.description} ({todo.agent})"
                    )
                    lines.append(line)
                    lines.extend(build_tree(todo.id, indent + 1))
            return lines

        tree_lines = ["# TODO リスト", ""]
        tree_lines.extend(build_tree(None))
        return "\n".join(tree_lines)

    def get_progress_report(self) -> Dict[str, Any]:
        """進捗レポートを取得"""
        total = len(self.todos)
        completed = sum(
            1 for t in self.todos.values() if t.status == TaskStatus.COMPLETED
        )
        in_progress = sum(
            1 for t in self.todos.values() if t.status == TaskStatus.IN_PROGRESS
        )
        pending = sum(1 for t in self.todos.values() if t.status == TaskStatus.PENDING)
        failed = sum(1 for t in self.todos.values() if t.status == TaskStatus.FAILED)

        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "failed": failed,
            "completion_rate": (completed / total * 100) if total > 0 else 0,
        }


# シングルトンインスタンス
todo_manager = TodoManager()
