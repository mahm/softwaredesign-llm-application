# プロンプトファイルを読み込む関数
import os


def load_prompt(name: str) -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.prompt")
    with open(prompt_path, "r") as f:
        return f.read()


def run_streaming_agent(agent, inputs, verbose=False):
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
