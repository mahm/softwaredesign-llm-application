"""メッセージ整形モジュール"""

from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class MessageFormatter:
    """メッセージ整形クラス"""

    def __init__(self):
        self.colors = {
            "user": "\033[96m",      # シアン
            "assistant": "\033[94m",  # 青
            "tool": "\033[93m",      # 黄
            "success": "\033[92m",   # 緑
            "error": "\033[91m",     # 赤
            "dim": "\033[2m",        # 薄い
            "bold": "\033[1m",       # 太字
            "reset": "\033[0m"       # リセット
        }

        # エージェント表示名
        self.agent_display_names = {
            "supervisor": "🎯 Supervisor",
            "task_decomposer": "📋 Task Decomposer",
            "research": "🔍 Research Agent",
            "writer": "✍️ Writer Agent",
            "__start__": "🚀 開始",
            "__end__": "🏁 終了"
        }

    def truncate_text(self, text: str, max_length: int = 150) -> str:
        """テキストを指定した長さに切り詰める"""
        if len(text) > max_length:
            return f"{text[:max_length]}..."
        return text

    def format_tool_use(self, content_item: dict) -> str:
        """ツール使用情報を整形"""
        tool_name = content_item.get("name", "")
        lines = [
            f"  {self.colors['tool']}→ ツール呼び出し: {tool_name}{self.colors['reset']}"]

        # ツールの引数を表示（デバッグ用）
        tool_input = content_item.get("input", {})
        if isinstance(tool_input, dict):
            for key, value in tool_input.items():
                value_str = str(value) if not isinstance(value, str) else value
                truncated_value = self.truncate_text(value_str, 80)
                lines.append(
                    f"    {self.colors['dim']}- {key}: {truncated_value}{self.colors['reset']}")

        return "\n".join(lines)

    def format_message(self, msg: Any) -> str:
        """メッセージを整形"""
        # HumanMessage
        if isinstance(msg, HumanMessage):
            return f"  {self.colors['user']}ユーザー入力: {msg.content}{self.colors['reset']}"

        # ToolMessage
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            truncated_content = self.truncate_text(content, 200)
            return f"  {self.colors['tool']}ツール結果: {truncated_content}{self.colors['reset']}"

        # AIMessage
        if isinstance(msg, AIMessage):
            return self._format_ai_message(msg)

        return f"  {self.colors['dim']}不明なメッセージ: {str(msg)[:100]}...{self.colors['reset']}"

    def _format_ai_message(self, msg: AIMessage) -> str:
        """AIメッセージを整形"""
        lines = []

        # AIMessageの内容がリストの場合（Anthropic形式）
        if isinstance(msg.content, list):
            for content_item in msg.content:
                if not isinstance(content_item, dict):
                    continue

                content_type = content_item.get("type")
                if content_type == "text":
                    text = content_item.get('text', '')
                    if text.strip():  # 空文字列でない場合のみ表示
                        truncated_text = self.truncate_text(text)
                        lines.append(
                            f"  {self.colors['assistant']}応答: {truncated_text}{self.colors['reset']}")
                elif content_type == "tool_use":
                    lines.append(self.format_tool_use(content_item))
        else:
            # AIMessageの内容が文字列の場合
            text = str(msg.content)
            if text.strip():  # 空文字列でない場合のみ表示
                truncated_text = self.truncate_text(text)
                lines.append(
                    f"  {self.colors['assistant']}応答: {truncated_text}{self.colors['reset']}")

            # tool_callsがある場合
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    lines.append(
                        f"  {self.colors['tool']}→ ツール呼び出し: {tool_call['name']}{self.colors['reset']}")

        return "\n".join(lines) if lines else ""

    def format_node_header(self, node_name: str, namespace: str = "") -> str:
        """ノードヘッダーを整形"""
        display_name = self.agent_display_names.get(node_name, node_name)

        if namespace:
            return f"\n{self.colors['bold']}📦 [サブグラフ: {namespace}]{self.colors['reset']}\n[{display_name}]"
        else:
            return f"\n{self.colors['bold']}[{display_name}]{self.colors['reset']}"

    def format_section_header(self, title: str, separator_char: str = "=", width: int = 60) -> str:
        """セクションヘッダーを整形"""
        return f"\n{self.colors['bold']}{separator_char * width}{self.colors['reset']}\n{self.colors['bold']}{title}{self.colors['reset']}\n{self.colors['bold']}{separator_char * width}{self.colors['reset']}"

    def format_completion_message(self, message: str) -> str:
        """完了メッセージを整形"""
        return f"\n{self.colors['success']}{self.colors['bold']}🎉 {message}{self.colors['reset']}"

    def format_error_message(self, message: str) -> str:
        """エラーメッセージを整形"""
        return f"\n{self.colors['error']}{self.colors['bold']}❌ {message}{self.colors['reset']}"

    def format_info_message(self, message: str) -> str:
        """情報メッセージを整形"""
        return f"{self.colors['dim']}{message}{self.colors['reset']}"
