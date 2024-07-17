import os
from typing import Any

from langgraph.graph.graph import CompiledGraph


# プロンプトファイルを読み込む関数
def load_prompt(name: str) -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.prompt")
    with open(prompt_path, "r") as f:
        return f.read()


# stream関数でエージェントを呼び出し、実行結果を逐次表示する
def run_streaming_agent(
    agent: CompiledGraph, inputs: list[dict[str, Any]], verbose: bool = False
) -> str:
    result = ""
    for s in agent.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            result = str(message)
        else:
            result = message.pretty_print()

        if verbose:
            display_result = ""
            if len(result) > 1000:
                display_result = result[:1000] + "..."
            else:
                display_result = result
            print(display_result)
    return result


# invoke関数でエージェントを呼び出し、実行結果を返す
def run_invoke_agent(agent: CompiledGraph, inputs: list[dict[str, Any]]) -> str:
    result = agent.invoke(inputs)
    return str(result["messages"][-1])
