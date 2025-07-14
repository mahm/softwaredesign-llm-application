"""進捗追跡モジュール"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from ..utils.memory import memory
from ..utils.todo_manager import TaskStatus


class ProgressTracker:
    """進捗追跡クラス"""

    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.task_start_times: Dict[str, datetime] = {}
        self.task_completion_times: Dict[str, datetime] = {}
        self.agent_activity: Dict[str, datetime] = {}

    def start_tracking(self):
        """追跡開始"""
        self.start_time = datetime.now()

    def track_task_start(self, task_id: str):
        """タスク開始を記録"""
        self.task_start_times[task_id] = datetime.now()

    def track_task_completion(self, task_id: str):
        """タスク完了を記録"""
        self.task_completion_times[task_id] = datetime.now()

    def track_agent_activity(self, agent_name: str):
        """エージェント活動を記録"""
        self.agent_activity[agent_name] = datetime.now()

    def get_elapsed_time(self) -> timedelta:
        """経過時間を取得"""
        if self.start_time is None:
            return timedelta(0)
        return datetime.now() - self.start_time

    def get_task_duration(self, task_id: str) -> Optional[timedelta]:
        """タスクの実行時間を取得"""
        start_time = self.task_start_times.get(task_id)
        completion_time = self.task_completion_times.get(task_id)

        if start_time and completion_time:
            return completion_time - start_time
        elif start_time:
            return datetime.now() - start_time

        return None

    def get_average_task_duration(self) -> Optional[timedelta]:
        """平均タスク実行時間を取得"""
        completed_durations = []

        for task_id in self.task_completion_times.keys():
            duration = self.get_task_duration(task_id)
            if duration:
                completed_durations.append(duration)

        if completed_durations:
            total_seconds = sum(d.total_seconds() for d in completed_durations)
            avg_seconds = total_seconds / len(completed_durations)
            return timedelta(seconds=avg_seconds)

        return None

    def estimate_remaining_time(self, pending_tasks: int) -> Optional[timedelta]:
        """残り時間を推定"""
        avg_duration = self.get_average_task_duration()
        if avg_duration and pending_tasks > 0:
            return avg_duration * pending_tasks
        return None

    def get_progress_stats(self) -> Dict[str, Any]:
        """進捗統計を取得"""
        todo_manager = memory.get("todo_manager")
        todos = {}

        if todo_manager and hasattr(todo_manager, "todos"):
            todos = {task_id: task.to_dict()
                     for task_id, task in todo_manager.todos.items()}

        total_tasks = len(todos)
        completed_tasks = sum(1 for task in todos.values()
                              if TaskStatus(task.get("status", "pending")) == TaskStatus.COMPLETED)
        in_progress_tasks = sum(1 for task in todos.values()
                                if TaskStatus(task.get("status", "pending")) == TaskStatus.IN_PROGRESS)
        pending_tasks = total_tasks - completed_tasks - in_progress_tasks

        completion_rate = (completed_tasks / total_tasks *
                           100) if total_tasks > 0 else 0

        elapsed = self.get_elapsed_time()
        avg_duration = self.get_average_task_duration()
        estimated_remaining = self.estimate_remaining_time(pending_tasks)

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": pending_tasks,
            "completion_rate": completion_rate,
            "elapsed_time": elapsed,
            "average_task_duration": avg_duration,
            "estimated_remaining_time": estimated_remaining,
            "last_activity": max(self.agent_activity.values()) if self.agent_activity else None
        }

    def format_time_display(self, td: timedelta) -> str:
        """時間を表示用にフォーマット"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def get_detailed_progress_report(self) -> str:
        """詳細な進捗レポートを生成"""
        stats = self.get_progress_stats()

        lines = [
            f"📊 詳細進捗レポート",
            f"",
            f"タスク状況:",
            f"  ✅ 完了: {stats['completed_tasks']}件",
            f"  🔄 進行中: {stats['in_progress_tasks']}件",
            f"  ⬜ 待機中: {stats['pending_tasks']}件",
            f"  📈 完了率: {stats['completion_rate']:.1f}%",
            f"",
            f"時間情報:",
            f"  ⏱️ 経過時間: {self.format_time_display(stats['elapsed_time'])}",
        ]

        if stats['average_task_duration']:
            lines.append(
                f"  ⏱️ 平均実行時間: {self.format_time_display(stats['average_task_duration'])}")

        if stats['estimated_remaining_time']:
            lines.append(
                f"  ⏱️ 推定残り時間: {self.format_time_display(stats['estimated_remaining_time'])}")

        if stats['last_activity']:
            last_activity_ago = datetime.now() - stats['last_activity']
            lines.append(
                f"  🕐 最終活動: {self.format_time_display(last_activity_ago)}前")

        return "\n".join(lines)


class PerformanceMonitor:
    """パフォーマンス監視クラス"""

    def __init__(self, progress_tracker: ProgressTracker):
        self.progress_tracker = progress_tracker
        self.monitoring = False
        self.alert_callbacks: List[Callable] = []

    def add_alert_callback(self, callback: Callable):
        """アラートコールバックを追加"""
        self.alert_callbacks.append(callback)

    async def start_monitoring(self, check_interval: float = 30.0):
        """パフォーマンス監視を開始"""
        self.monitoring = True

        while self.monitoring:
            await self._check_performance()
            await asyncio.sleep(check_interval)

    def stop_monitoring(self):
        """監視を停止"""
        self.monitoring = False

    async def _check_performance(self):
        """パフォーマンスをチェック"""
        stats = self.progress_tracker.get_progress_stats()

        # 長時間実行アラート
        if stats['elapsed_time'].total_seconds() > 1800:  # 30分
            await self._trigger_alert("⚠️ 実行時間が30分を超えました")

        # 停滞アラート
        if stats['last_activity']:
            inactive_time = datetime.now() - stats['last_activity']
            if inactive_time.total_seconds() > 300:  # 5分
                await self._trigger_alert("⚠️ 5分間活動がありません")

        # 低完了率アラート（実行開始から10分後以降）
        if (stats['elapsed_time'].total_seconds() > 600 and
                stats['completion_rate'] < 20):
            await self._trigger_alert("⚠️ 完了率が低い状態が続いています")

    async def _trigger_alert(self, message: str):
        """アラートを発火"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                print(f"アラートコールバックエラー: {e}")
