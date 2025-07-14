"""Claude Code風タスクリスト表示エンジン"""

import asyncio
from datetime import datetime
from typing import Dict, Any
from ..utils.todo_manager import TaskStatus


class TaskDisplayEngine:
    """タスクリスト表示エンジン"""

    def __init__(self):
        self.status_icons = {
            TaskStatus.PENDING: "⬜",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌"
        }

        self.colors = {
            "success": "\033[92m",
            "warning": "\033[93m",
            "error": "\033[91m",
            "info": "\033[94m",
            "dim": "\033[2m",
            "bold": "\033[1m",
            "reset": "\033[0m"
        }

        self.agent_icons = {
            "task_decomposer": "📋",
            "research": "🔍",
            "writer": "✍️",
            "supervisor": "🎯"
        }

        self.last_displayed_state = None
        self.start_time = datetime.now()

    def render_progress_bar(self, completed: int, total: int, width: int = 20) -> str:
        """進捗バーを描画"""
        if total == 0:
            return f"[{'░' * width}] 0% (0/0)"

        progress = completed / total
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(progress * 100)

        return f"[{bar}] {percentage}% ({completed}/{total})"

    def format_task_line(self, task: Dict[str, Any], indent: int = 0) -> str:
        """単一タスクの表示行を整形"""
        status = TaskStatus(task.get("status", "pending"))
        icon = self.status_icons[status]
        task_id = task.get("id", "UNKNOWN")
        description = task.get("description", "")
        agent = task.get("agent", "")

        # 色分け
        if status == TaskStatus.COMPLETED:
            color = self.colors["success"]
        elif status == TaskStatus.FAILED:
            color = self.colors["error"]
        elif status == TaskStatus.IN_PROGRESS:
            color = self.colors["info"]
        else:
            color = self.colors["dim"]

        # エージェントアイコン
        agent_icon = self.agent_icons.get(agent, "🤖")

        # 実行時間表示（進行中タスク用）
        time_info = ""
        if status == TaskStatus.IN_PROGRESS:
            elapsed = datetime.now() - datetime.fromisoformat(task.get("updated_at",
                                                                       datetime.now().isoformat()))
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            time_info = f" ⏱️ {minutes}m {seconds}s"
        elif status == TaskStatus.COMPLETED:
            time_info = " ✓"

        indent_str = "  " * indent
        line = f"{color}{indent_str}{icon} [{task_id}] {description} {agent_icon} ({agent}){time_info}{self.colors['reset']}"

        return line

    def render_task_list(self, todos: Dict[str, Any]) -> str:
        """タスクリスト全体を描画"""
        if not todos:
            return f"{self.colors['dim']}📋 タスクリスト: (タスクはまだ作成されていません){self.colors['reset']}"

        # 統計情報を計算
        total_tasks = len(todos)
        completed_tasks = sum(1 for task in todos.values()
                              if TaskStatus(task.get("status", "pending")) == TaskStatus.COMPLETED)
        in_progress_tasks = sum(1 for task in todos.values()
                                if TaskStatus(task.get("status", "pending")) == TaskStatus.IN_PROGRESS)
        failed_tasks = sum(1 for task in todos.values()
                           if TaskStatus(task.get("status", "pending")) == TaskStatus.FAILED)

        # ヘッダー
        progress_bar = self.render_progress_bar(completed_tasks, total_tasks)
        elapsed_time = datetime.now() - self.start_time
        elapsed_minutes = int(elapsed_time.total_seconds() // 60)
        elapsed_seconds = int(elapsed_time.total_seconds() % 60)

        lines = [
            f"\n{self.colors['bold']}📋 タスク実行状況{self.colors['reset']} {progress_bar}",
            f"{self.colors['dim']}実行時間: {elapsed_minutes}m {elapsed_seconds}s{self.colors['reset']}"
        ]

        if failed_tasks > 0:
            lines.append(
                f"{self.colors['error']}⚠️ 失敗: {failed_tasks}件{self.colors['reset']}")

        lines.append("")  # 空行

        # タスク一覧を表示（IDでソート）
        sorted_tasks = sorted(todos.items(), key=lambda x: x[1].get("id", ""))

        for task_id, task_data in sorted_tasks:
            lines.append(self.format_task_line(task_data))

        # 進行中タスクがある場合、現在の活動を表示
        if in_progress_tasks > 0:
            lines.append("")
            lines.append(
                f"{self.colors['info']}🔄 実行中: {in_progress_tasks}件のタスクが進行中{self.colors['reset']}")

        return "\n".join(lines)

    def get_current_todos(self) -> Dict[str, Any]:
        """現在のTODOリストを取得"""
        try:
            # 直接todo_managerを参照
            from ..utils.todo_manager import todo_manager
            return {task_id: task.to_dict() for task_id, task in todo_manager.todos.items()}
        except Exception:
            pass
        return {}

    def has_todos_changed(self) -> bool:
        """TODOリストが変更されたかをチェック"""
        current_todos = self.get_current_todos()
        changed = current_todos != self.last_displayed_state
        if changed:
            self.last_displayed_state = current_todos.copy()
        return changed

    def render_current_status(self) -> str:
        """現在のタスク状況を描画"""
        todos = self.get_current_todos()
        return self.render_task_list(todos)

    def should_display_update(self) -> bool:
        """表示更新が必要かを判定"""
        return self.has_todos_changed()


class TaskMonitor:
    """タスク状態監視クラス"""

    def __init__(self, display_engine: TaskDisplayEngine):
        self.display_engine = display_engine
        self.is_monitoring = False
        self.update_interval = 1.0  # 1秒間隔

    async def start_monitoring(self, callback=None):
        """タスク監視を開始"""
        self.is_monitoring = True

        while self.is_monitoring:
            if self.display_engine.should_display_update():
                status_display = self.display_engine.render_current_status()
                if callback:
                    await callback(status_display)
                else:
                    print(f"\r{' ' * 100}\r", end="")  # 行クリア
                    print(status_display)

            await asyncio.sleep(self.update_interval)

    def stop_monitoring(self):
        """タスク監視を停止"""
        self.is_monitoring = False
