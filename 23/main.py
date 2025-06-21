"""グラフの動作確認用スクリプト"""

import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from src.sd_23.supervisor_graph import create_supervisor_workflow
from src.sd_23.swarm_graph import create_swarm_workflow
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()


def format_message(msg: Any, is_swarm: bool = False) -> None:
    """メッセージを整形して表示"""
    if isinstance(msg, ToolMessage):
        print(f"  ツール結果: {msg.content}")
        return

    if not isinstance(msg, AIMessage):
        return

    # AIメッセージの処理
    if isinstance(msg.content, list):
        for item in msg.content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    print(f"  応答: {item.get('text', '')}")
                elif item.get("type") == "tool_use":
                    _print_tool_use(item.get("name", ""), is_swarm)
    else:
        print(f"  応答: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                _print_tool_use(tool_call["name"], is_swarm)


def _print_tool_use(tool_name: str, is_swarm: bool) -> None:
    """ツール使用情報を表示"""
    if is_swarm and ("transfer_to" in tool_name or "handoff" in tool_name):
        print(f"  → ハンドオフ: {tool_name}")
    else:
        print(f"  → ツール呼び出し: {tool_name}")


async def stream_graph_output(
    app: Any, input_text: str, config: Optional[Dict] = None, is_swarm: bool = False
) -> None:
    """グラフの出力をストリーミング表示"""
    input_data = {"messages": [HumanMessage(content=input_text)]}
    seen_messages = set()

    stream_args = [input_data]
    if config:
        stream_args.append(config)

    async for chunk in app.astream(*stream_args, stream_mode="updates", subgraphs=True):
        if isinstance(chunk, tuple):
            parent_ns, outputs = chunk
            # サブグラフの場合はエージェント名とIDを含む
            if parent_ns and len(parent_ns) > 0:
                parent_info = (
                    parent_ns[0] if isinstance(parent_ns[0], str) else str(parent_ns[0])
                )
                # IDを除去してエージェント名のみ抽出
                parent_name = (
                    parent_info.split(":")[0] if ":" in parent_info else parent_info
                )
            else:
                parent_name = None
            for node_name, output in outputs.items():
                _print_node(node_name, output, is_swarm, seen_messages, parent_name)
        else:
            for node_name, output in chunk.items():
                _print_node(node_name, output, is_swarm, seen_messages)


def _print_node(
    node_name: str,
    output: Dict[str, Any],
    is_swarm: bool,
    seen_messages: set,
    parent_name: Optional[str] = None,
) -> None:
    """ノード出力を表示"""
    label = f"[{parent_name}:{node_name}]" if parent_name else f"[{node_name}]"
    print(label)

    if "messages" in output:
        for msg in output["messages"]:
            msg_id = getattr(msg, "id", None) or str(hash(str(msg.content)))
            if msg_id not in seen_messages:
                seen_messages.add(msg_id)
                format_message(msg, is_swarm)

    if "active_agent" in output:
        print(f"  アクティブエージェント: {output['active_agent']}")

    print()


async def run_pattern_test(
    pattern_name: str,
    create_workflow_fn,
    test_description: str,
    test_input: str,
    thread_id: str,
    is_swarm: bool = False,
) -> None:
    """パターンのテストを実行"""
    print(f"=== {pattern_name} Pattern Test ===\n")
    print(f"--- {test_description} ---")
    print(f"入力: {test_input}\n")

    workflow = create_workflow_fn()
    app = workflow.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}

    await stream_graph_output(app, test_input, config, is_swarm)


async def main():
    """メイン関数"""
    import sys

    # コマンドライン引数でパターンを選択
    pattern = sys.argv[1] if len(sys.argv) > 1 else "all"

    try:
        if pattern in ["all", "supervisor"]:
            # Supervisorパターンのテスト
            await run_pattern_test(
                "Supervisor",
                create_supervisor_workflow,
                "Supervisorパターンのテスト",
                "最新のAI技術について調査し、その市場規模（2025年の予測値）を調べてください。"
                "その後、年平均成長率を25%として、5年後の市場規模を計算してください。",
                "supervisor_test",
            )

            if pattern == "all":
                print()  # パターン間の区切り

        if pattern in ["all", "swarm"]:
            # Swarmパターンのテスト
            await run_pattern_test(
                "Swarm",
                create_swarm_workflow,
                "Swarmパターンのテスト",
                "エラーコード500が出ています。どういう意味でしょうか？"
                "また、このエラーの詳細な診断と解決方法を教えてください。",
                "swarm_test",
                is_swarm=True,
            )

        print("\n✅ テストが完了しました！")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print("Anthropic APIキーが設定されているか確認してください。")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
