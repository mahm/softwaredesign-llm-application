from functools import lru_cache
from typing import Annotated, ClassVar, Literal, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


# グラフのステートを定義
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# グラフの設定を定義
class GraphConfig(BaseModel):
    ANTHROPIC: ClassVar[str] = "anthropic"
    OPENAI: ClassVar[str] = "openai"

    model_name: Literal["anthropic", "openai"]


# モデルを取得する関数
@lru_cache(maxsize=2)
def _get_model(model_name: str) -> ChatOpenAI | ChatAnthropic:
    if model_name == GraphConfig.OPENAI:
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == GraphConfig.ANTHROPIC:
        model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20241022")
    else:
        raise ValueError(f"サポートしていないモデルです: {model_name}")

    model = model.bind_tools(tools)
    return model


# ツール実行のためのノードを定義
tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)


# ツール実行の継続条件を定義
def should_continue(state: AgentState) -> Literal["end", "continue"]:
    last_message = state.messages[-1]
    return "continue" if last_message.tool_calls else "end"


# モデルを実行する関数
def call_model(state: AgentState, config: GraphConfig) -> dict:
    messages = state.messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ] + messages
    # configの情報からモデル名を取得
    model_name = config.get("configurable", {}).get("model_name", "anthropic")
    # モデル名から利用するモデルのインスタンスを取得
    model = _get_model(model_name)
    # モデルを実行
    response = model.invoke(messages)
    return {"messages": [response]}


# グラフを定義
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# ノードを追加
workflow.add_node("call_model", call_model)
workflow.add_node("tool_node", tool_node)

# エッジを追加
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    },
)
workflow.add_edge("tool_node", "call_model")

graph = workflow.compile()
