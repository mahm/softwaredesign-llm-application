"""統合ターミナルUI制御"""

import asyncio
import os
from typing import Dict, Any, Optional, Set

from .message_formatter import MessageFormatter
from .task_display import TaskDisplayEngine, TaskMonitor


class TerminalUI:
    """統合ターミナルUI制御クラス"""

    def __init__(self):
        self.formatter = MessageFormatter()
        self.task_display = TaskDisplayEngine()
        self.task_monitor = TaskMonitor(self.task_display)
        self.seen_messages: Set = set()
        self.is_debug_mode = False

        # 表示設定
        self.show_task_progress = False  # 一時的に無効化
        self.show_subgraph_details = True
        self.auto_clear_screen = False

    def clear_screen(self):
        """画面クリア"""
        if self.auto_clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')

    def print_startup_banner(self, debug_mode: bool = False):
        """起動時のバナー表示"""
        self.is_debug_mode = debug_mode

        print(self.formatter.format_section_header("文章執筆支援システム 動作確認"))

        if debug_mode:
            print(self.formatter.format_info_message("\n🐛 デバッグモードで実行中..."))
            print(self.formatter.format_info_message(
                "(通常モードで実行するには、--debugオプションを外してください)"))
        else:
            print(self.formatter.format_info_message("\n📋 通常モードで実行中..."))
            print(self.formatter.format_info_message(
                "(詳細なデバッグ情報を見るには、--debugオプションを付けてください)"))
            print(self.formatter.format_info_message(
                "例: uv run python main.py --debug"))

        print(self.formatter.format_info_message(
            "\n✍️ Writer実装: create_react_agent版（小さなツールの組み合わせ）"))

    def print_test_header(self, test_name: str, input_text: str):
        """テストヘッダー表示"""
        print(self.formatter.format_section_header(f"テスト: {test_name}"))
        print(f"入力: {input_text}\n")

    def print_node_output(self, node_name: str, node_output: Dict[str, Any], namespace: str = ""):
        """ノード出力を表示"""
        # ヘッダー表示
        print(self.formatter.format_node_header(node_name, namespace))

        # メッセージ処理
        if "messages" in node_output:
            for msg in node_output["messages"]:
                # メッセージの重複チェック
                msg_id = getattr(msg, "id", None) or str(
                    hash(str(msg.content)))
                if msg_id not in self.seen_messages:
                    self.seen_messages.add(msg_id)
                    formatted_message = self.formatter.format_message(msg)
                    if formatted_message:  # 空文字列でない場合のみ表示
                        print(formatted_message)

        # その他の状態情報
        for key, value in node_output.items():
            if key != "messages":
                print(f"  {key}: {value}")

    async def run_with_task_monitoring(
        self,
        app: Any,
        input_data: Dict[str, Any],
        config: Optional[Dict] = None,
        test_name: str = "",
        input_text: str = ""
    ):
        """タスク監視付きで実行"""
        self.print_test_header(test_name, input_text)

        # タスク監視を開始
        if self.show_task_progress:
            monitoring_task = asyncio.create_task(
                self.task_monitor.start_monitoring(self._on_task_update)
            )

        try:
            # メイン実行
            execution_task = asyncio.create_task(
                self._run_main_execution(app, input_data, config)
            )

            # 実行完了まで待機
            await execution_task

        finally:
            # タスク監視を停止
            if self.show_task_progress:
                self.task_monitor.stop_monitoring()
                try:
                    await asyncio.wait_for(monitoring_task, timeout=1.0)
                except asyncio.TimeoutError:
                    monitoring_task.cancel()

        # 最終タスク状況表示
        if self.show_task_progress:
            final_status = self.task_display.render_current_status()
            print(final_status)

        print(self.formatter.format_completion_message("テスト完了"))

    async def _run_main_execution(self, app: Any, input_data: Dict[str, Any], config: Optional[Dict]):
        """メイン実行処理"""
        stream_config = {
            "stream_mode": "updates",
            "subgraphs": True
        }

        if config:
            async for chunk in app.astream(input_data, config, **stream_config):
                self._process_chunk(chunk)
        else:
            async for chunk in app.astream(input_data, **stream_config):
                self._process_chunk(chunk)

    def _process_chunk(self, chunk: Any):
        """チャンクデータを処理"""
        if isinstance(chunk, tuple):
            # サブグラフの出力
            namespace, output = chunk
            if self.show_subgraph_details:
                namespace_display = f"サブグラフ: {namespace}" if namespace else ""
                for node_name, node_output in output.items():
                    self.print_node_output(
                        node_name, node_output, namespace_display)
        else:
            # メイングラフの出力
            for node_name, node_output in chunk.items():
                self.print_node_output(node_name, node_output)

    async def _on_task_update(self, status_display: str):
        """タスク更新時のコールバック"""
        # 現在の表示領域を一時的にクリアして、タスク状況を上部に表示
        # この実装では、タスク状況が更新された時にのみ表示
        pass  # 現在は何もしない（将来的にリアルタイム更新を実装）

    async def run_debug_mode(
        self,
        app: Any,
        input_data: Dict[str, Any],
        config: Optional[Dict] = None,
        test_name: str = "",
        input_text: str = ""
    ):
        """デバッグモードで実行"""
        self.print_test_header(f"🐛 デバッグモード: {test_name}", input_text)

        print(self.formatter.format_info_message("📊 イベントストリーム開始...\n"))

        event_config = config or {}

        try:
            async for event in app.astream_events(input_data, version="v2", config=event_config):
                self._process_debug_event(event)
        except Exception as e:
            print(self.formatter.format_error_message(f"デバッグ実行エラー: {e}"))
            raise

        print(self.formatter.format_completion_message("デバッグ完了"))

    def _process_debug_event(self, event: Dict[str, Any]):
        """デバッグイベントを処理"""
        event_type = event.get("event", "")

        if event_type == "on_chain_start":
            chain_name = event.get("name", "")
            print(f"🔗 チェーン開始: {chain_name}")

        elif event_type == "on_chain_end":
            chain_name = event.get("name", "")
            print(f"✔️ チェーン終了: {chain_name}")

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            print(f"🔧 ツール開始: {tool_name}")

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "")
            print(f"✔️ ツール終了: {tool_name}")

        elif event_type == "on_chat_model_stream":
            data = event.get("data", {})
            chunk = data.get("chunk", None)
            if chunk:
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")

                if content:
                    truncated_content = self.formatter.truncate_text(
                        str(content), 100)
                    print(f"💬 LLM出力: {truncated_content}")

        elif event_type == "on_chat_model_start":
            model_name = event.get("name", "")
            print(f"🤖 モデル開始: {model_name}")

    def print_completion_summary(self):
        """完了時のサマリー表示"""
        print(self.formatter.format_section_header("🎉 すべてのテストが完了しました！"))

    def print_error_summary(self, error: Exception):
        """エラー時のサマリー表示"""
        print(self.formatter.format_error_message(f"システムエラーが発生しました: {error}"))
        print("\n以下を確認してください:")
        print("1. ANTHROPIC_API_KEYが正しく設定されているか")
        print("2. TAVILY_API_KEYが正しく設定されているか")
        print("3. 必要なパッケージがインストールされているか (uv sync)")

    def configure(
        self,
        show_task_progress: bool = True,
        show_subgraph_details: bool = True,
        auto_clear_screen: bool = False
    ):
        """表示設定"""
        self.show_task_progress = show_task_progress
        self.show_subgraph_details = show_subgraph_details
        self.auto_clear_screen = auto_clear_screen
