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
    """計算エージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")  # type: ignore[call-arg]

    tools = [add, multiply, divide]

    prompt = """あなたは計算専門のエージェントです。

役割と制約：
1. 与えられた数値と計算式に対して、ツールを使用して計算のみ実行
2. 情報収集・調査・推論は一切行わない
3. 計算手順を最小限に示し、最終結果を明確に報告

出力形式：
- 「計算実行：[計算式]」
- 「結果：[数値]」"""

    agent = create_react_agent(
        model=model, tools=tools, name="math_expert", prompt=prompt
    )

    return agent
