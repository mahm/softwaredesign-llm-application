"""グラフの動作確認用スクリプト"""

import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from src.sd_23.supervisor_graph import create_supervisor_graph
from src.sd_23.swarm_graph import create_swarm_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 環境変数を読み込み
load_dotenv()


def print_message_info(msg: Any, is_swarm: bool = False) -> None:
    """メッセージの情報を表示"""
    if isinstance(msg, AIMessage):
        # Anthropicのメッセージ形式を処理
        if isinstance(msg.content, list):
            for content_item in msg.content:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "text":
                        print(f"  応答: {content_item.get('text', '')}")
                    elif content_item.get("type") == "tool_use":
                        tool_name = content_item.get("name", "")
                        if is_swarm and (
                            "transfer_to" in tool_name or "handoff" in tool_name
                        ):
                            print(f"  → ハンドオフ: {tool_name}")
                        else:
                            print(f"  → ツール呼び出し: {tool_name}")
        else:
            # content がリストでない場合は、従来の処理
            print(f"  応答: {msg.content}")
            # tool_callsがある場合のみ処理（contentがリストでない場合のみ）
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]
                    if is_swarm and (
                        "transfer_to" in tool_name or "handoff" in tool_name
                    ):
                        print(f"  → ハンドオフ: {tool_name}")
                    else:
                        print(f"  → ツール呼び出し: {tool_name}")
    elif isinstance(msg, ToolMessage):
        print(f"  ツール結果: {msg.content}")


def print_node_output(
    node_name: str,
    node_output: Dict[str, Any],
    is_swarm: bool = False,
    seen_messages: set = None,
) -> None:
    """ノードの出力を表示"""
    if seen_messages is None:
        seen_messages = set()

    print(f"[{node_name}]")

    if "messages" in node_output:
        for msg in node_output["messages"]:
            # メッセージのIDまたは内容でユニーク性を判定
            msg_id = getattr(msg, "id", None) or str(hash(str(msg.content)))
            if msg_id not in seen_messages:
                seen_messages.add(msg_id)
                print_message_info(msg, is_swarm)

    if "active_agent" in node_output:
        print(f"  アクティブエージェント: {node_output['active_agent']}")

    print()


async def run_test(
    app: Any,
    test_name: str,
    input_text: str,
    config: Optional[Dict] = None,
    is_swarm: bool = False,
) -> None:
    """テストを実行して結果を表示"""
    print(f"--- {test_name} ---")
    print(f"入力: {input_text}\n")

    input_data = {"messages": [HumanMessage(content=input_text)]}
    seen_messages = set()  # メッセージの重複を防ぐ

    if config:
        async for chunk in app.astream(input_data, config):
            for node_name, node_output in chunk.items():
                print_node_output(node_name, node_output, is_swarm, seen_messages)
    else:
        async for chunk in app.astream(input_data):
            for node_name, node_output in chunk.items():
                print_node_output(node_name, node_output, is_swarm, seen_messages)


async def test_supervisor():
    """Supervisorパターンのテスト"""
    print("=== Supervisor Pattern Test ===\n")
    # ワークフローを取得してテスト用にCheckpointerを追加
    from src.sd_23.supervisor_graph import create_supervisor_workflow

    workflow = create_supervisor_workflow()
    app = workflow.compile(checkpointer=InMemorySaver())

    # テストケース
    tests = [
        ("数学タスクのテスト", "10と20を足して、その結果に5を掛けてください"),
        ("調査タスクのテスト", "量子コンピュータについて調べてください"),
    ]

    # Checkpointerを使用しているため、configが必要
    config = {"configurable": {"thread_id": "supervisor_test"}}

    for i, (test_name, input_text) in enumerate(tests):
        await run_test(app, test_name, input_text, config)
        # 最後のテスト以外の後に空行を追加
        if i < len(tests) - 1:
            print()


async def test_swarm():
    """Swarmパターンのテスト"""
    print("\n=== Swarm Pattern Test ===\n")
    # ワークフローを取得してテスト用にCheckpointerを追加
    from src.sd_23.swarm_graph import create_swarm_workflow

    workflow = create_swarm_workflow()
    app = workflow.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "test123"}}

    # テストケース
    tests = [
        ("FAQタスクのテスト", "パスワードのリセット方法を教えてください"),
        (
            "技術サポートへのハンドオフテスト",
            "システムエラーが発生していて、詳細な診断が必要です",
        ),
    ]

    for i, (test_name, input_text) in enumerate(tests):
        await run_test(app, test_name, input_text, config, is_swarm=True)
        # 最後のテスト以外の後に空行を追加
        if i < len(tests) - 1:
            print()


async def main():
    """メイン関数"""
    try:
        await test_supervisor()
        await test_swarm()
        print("\n✅ すべてのテストが完了しました！")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print("Anthropic APIキーが設定されているか確認してください。")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
