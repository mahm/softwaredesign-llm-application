"""文章執筆支援システムの動作確認用スクリプト"""

import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.sd_24.main import create_writing_assistant_workflow
import traceback

# 環境変数を読み込み
load_dotenv()


def truncate_text(text: str, max_length: int = 200) -> str:
    """長いテキストを省略表示用に切り詰める"""
    if len(text) > max_length:
        return f"{text[:max_length]}..."
    return text


def print_tool_use(content_item: dict) -> None:
    """ツール使用情報を表示"""
    tool_name = content_item.get("name", "")
    print(f"  → ツール呼び出し: {tool_name}")

    # ツールの引数を表示（デバッグ用）
    tool_input = content_item.get("input", {})
    if not isinstance(tool_input, dict):
        return

    for key, value in tool_input.items():
        value_str = str(value) if not isinstance(value, str) else value
        print(f"    - {key}: {truncate_text(value_str, 100)}")


def print_message_info(msg: Any) -> None:
    """メッセージの情報を表示"""
    # HumanMessage
    if isinstance(msg, HumanMessage):
        print(f"  ユーザー入力: {msg.content}")
        return

    # ToolMessage
    if isinstance(msg, ToolMessage):
        content = str(msg.content)
        print(f"  ツール結果: {truncate_text(content, 150)}")
        return

    # AIMessage
    if not isinstance(msg, AIMessage):
        return

    # AIMessageの内容がリストの場合（Anthropic形式）
    if isinstance(msg.content, list):
        for content_item in msg.content:
            if not isinstance(content_item, dict):
                continue

            content_type = content_item.get("type")
            if content_type == "text":
                text = content_item.get('text', '')
                print(f"  応答: {truncate_text(text)}")
            elif content_type == "tool_use":
                print_tool_use(content_item)
    else:
        # AIMessageの内容が文字列の場合
        text = str(msg.content)
        print(f"  応答: {truncate_text(text)}")

        # tool_callsがある場合
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"  → ツール呼び出し: {tool_call['name']}")


def print_node_output(
    node_name: str,
    node_output: Dict[str, Any],
    seen_messages: set | None = None,
) -> None:
    """ノードの出力を表示"""
    if seen_messages is None:
        seen_messages = set()

    # ノード名を色付きで表示（ターミナルで見やすくするため）
    node_display_names = {
        "supervisor": "🎯 Supervisor",
        "task_decomposer": "📋 Task Decomposer",
        "research": "🔍 Research Agent",
        "writer": "✍️  Writer Agent (小さなツールの組み合わせ版)",
        "__start__": "🚀 開始",
        "__end__": "🏁 終了"
    }

    display_name = node_display_names.get(node_name, node_name)
    print(f"\n[{display_name}]")

    if "messages" in node_output:
        for msg in node_output["messages"]:
            # メッセージのIDまたは内容でユニーク性を判定
            msg_id = getattr(msg, "id", None) or str(hash(str(msg.content)))
            if msg_id not in seen_messages:
                seen_messages.add(msg_id)
                print_message_info(msg)

    # その他の状態情報があれば表示
    for key, value in node_output.items():
        if key != "messages":
            print(f"  {key}: {value}")


async def run_test(
    app: Any,
    test_name: str,
    input_text: str,
    config: Optional[Dict] = None,
) -> None:
    """テストを実行して結果を表示（サブグラフ対応）"""
    print(f"\n{'='*60}")
    print(f"テスト: {test_name}")
    print(f"{'='*60}")
    print(f"入力: {input_text}\n")

    input_data = {"messages": [HumanMessage(content=input_text)]}
    seen_messages: set = set()  # メッセージの重複を防ぐ

    try:
        # subgraphs=Trueを使用して、サブグラフの実行状況も取得
        if config:
            async for chunk in app.astream(input_data, config, stream_mode="updates", subgraphs=True):
                if isinstance(chunk, tuple):
                    # サブグラフ（サブエージェント）の出力
                    namespace, output = chunk
                    print(f"\n📦 [サブグラフ: {namespace}]")
                    for node_name, node_output in output.items():
                        print_node_output(
                            node_name, node_output, seen_messages)
                else:
                    # メイングラフ（supervisor）の出力
                    for node_name, node_output in chunk.items():
                        print_node_output(
                            node_name, node_output, seen_messages)
        else:
            async for chunk in app.astream(input_data, stream_mode="updates", subgraphs=True):
                if isinstance(chunk, tuple):
                    namespace, output = chunk
                    print(f"\n📦 [サブグラフ: {namespace}]")
                    for node_name, node_output in output.items():
                        print_node_output(
                            node_name, node_output, seen_messages)
                else:
                    for node_name, node_output in chunk.items():
                        print_node_output(
                            node_name, node_output, seen_messages)

        print(f"\n{'='*60}")
        print("✅ テスト完了")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        traceback.print_exc()


async def run_test_with_debug(
    app: Any,
    test_name: str,
    input_text: str,
    config: Optional[Dict] = None,
) -> None:
    """デバッグモードでテストを実行（イベントレベルの詳細表示）"""
    print(f"\n{'='*60}")
    print(f"🐛 デバッグモード: {test_name}")
    print(f"{'='*60}")
    print(f"入力: {input_text}\n")

    input_data = {"messages": [HumanMessage(content=input_text)]}

    try:
        # astream_eventsを使用して、イベントレベルの詳細を取得
        print("📊 イベントストリーム開始...\n")

        event_config = config or {}
        async for event in app.astream_events(input_data, version="v2", config=event_config):
            event_type = event.get("event", "")

            if event_type == "on_chain_start":
                chain_name = event.get("name", "")
                print(f"🔗 チェーン開始: {chain_name}")

            elif event_type == "on_chain_end":
                chain_name = event.get("name", "")
                print(f"✔️  チェーン終了: {chain_name}")

            elif event_type == "on_tool_start":
                tool_name = event.get("name", "")
                print(f"🔧 ツール開始: {tool_name}")

            elif event_type == "on_tool_end":
                tool_name = event.get("name", "")
                print(f"✔️  ツール終了: {tool_name}")

            elif event_type == "on_chat_model_stream":
                data = event.get("data", {})
                chunk = data.get("chunk", None)
                if chunk:
                    # AIMessageChunkオブジェクトの場合は直接contentにアクセス
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    else:
                        # dictの場合
                        content = chunk.get("content", "") if isinstance(
                            chunk, dict) else ""
                    if content:
                        print(f"💬 LLM出力: {truncate_text(content, 100)}")

            elif event_type == "on_chat_model_start":
                model_name = event.get("name", "")
                print(f"🤖 モデル開始: {model_name}")

        print(f"\n{'='*60}")
        print("✅ デバッグ完了")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        traceback.print_exc()


async def main():
    """メイン関数"""
    print("文章執筆支援システム 動作確認")
    print("="*60)

    # コマンドライン引数でデバッグモードを選択可能にする
    import sys
    debug_mode = "--debug" in sys.argv

    try:
        # ワークフローを取得してテスト用にCheckpointerを追加
        workflow = create_writing_assistant_workflow()
        app = workflow.compile(checkpointer=InMemorySaver())

        # テストケース
        tests = [
            (
                "最新情報レポート",
                "2025年5月に行われたLangChain Interruptについてレポートしてください。また、日本の有力な情報源についても挙げてください。"
            )
        ]

        # 使用方法の表示
        if debug_mode:
            print("\n🐛 デバッグモードで実行中...")
            print("(通常モードで実行するには、--debugオプションを外してください)\n")
        else:
            print("\n📋 通常モードで実行中...")
            print("(詳細なデバッグ情報を見るには、--debugオプションを付けてください)")
            print("例: uv run python main.py --debug\n")

        # Checkpointerを使用しているため、configが必要
        for i, (test_name, input_text) in enumerate(tests):
            config = {
                "configurable": {"thread_id": f"test_{i+1}"},
                "recursion_limit": 100
            }

            if debug_mode:
                await run_test_with_debug(app, test_name, input_text, config)
            else:
                await run_test(app, test_name, input_text, config)

            # テスト間に区切りを入れる
            if i < len(tests) - 1:
                print("\n" + "="*60 + "\n")
                print("次のテストを開始します...")
                await asyncio.sleep(2)  # 見やすさのため少し待機

        print("\n" + "="*60)
        print("🎉 すべてのテストが完了しました！")
        print("="*60)

    except Exception as e:
        print(f"\n❌ システムエラーが発生しました: {e}")
        print("\n以下を確認してください:")
        print("1. ANTHROPIC_API_KEYが正しく設定されているか")
        print("2. TAVILY_API_KEYが正しく設定されているか")
        print("3. 必要なパッケージがインストールされているか (uv sync)")
        print("\n詳細なエラー情報:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
