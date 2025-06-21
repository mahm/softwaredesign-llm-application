"""数学計算に特化したエージェント"""

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


@tool
def add(a: float, b: float) -> float:
    """2つの数値を加算します"""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """2つの数値を乗算します"""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """2つの数値を除算します（bがゼロでない場合）"""
    if b == 0:
        return float("inf")
    return a / b


def create_math_agent() -> CompiledGraph:
    """数学エージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")  # type: ignore[call-arg]

    tools = [add, multiply, divide]

    prompt = """あなたは数学計算の専門家です。
与えられた数学的な問題を正確に解決してください。
計算の過程を説明しながら、ツールを使用して正確な答えを提供してください。"""

    agent = create_react_agent(
        model=model, tools=tools, name="math_expert", prompt=prompt
    )

    return agent
