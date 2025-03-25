import asyncio
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from src.sd_20.mcp_manager import load_all_mcp_tools
from src.sd_20.state import CustomAgentState

# 環境変数の読み込み
load_dotenv()


def create_agent():
    # ツール（MCPツール）の読み込み
    tools = asyncio.run(load_all_mcp_tools())

    # ツールの説明の作成
    tool_descriptions = "\n\n".join(
        [f"### {tool.name}\n{tool.description}" for tool in tools]
    )

    # 本日の日付の取得
    current_date = datetime.now().strftime("%Y年%m月%d日")

    # プロンプトの読み込み
    with open("src/sd_20/prompts/system.txt", "r") as f:
        prompt = f.read()

    # プロンプトの作成
    prompt = prompt.format(
        tool_descriptions=tool_descriptions,
        current_date=current_date,
    )

    # モデルの設定
    model = ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        timeout=None,
        stop=None,
        max_tokens=4_096,
    )

    # エージェントの作成
    graph = create_react_agent(
        model,
        tools=tools,
        prompt=prompt,
        state_schema=CustomAgentState,
        checkpointer=MemorySaver(),
    )

    return graph


graph = create_agent()
