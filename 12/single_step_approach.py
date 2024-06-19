import argparse
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from settings import Settings
from tools import search, report_writer
from utility import load_prompt

settings = Settings()


def single_step_agent():
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    tools = [search, report_writer]
    return create_react_agent(
        llm, tools=tools, messages_modifier=load_prompt("single_step_answering_system")
    )


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Process some queries.")
    parser.add_argument("--task", type=str, required=True, help="The query to search")

    # コマンドライン引数を解析
    args = parser.parse_args()

    inputs = {"messages": [("user", args.task)]}
    for s in single_step_agent().stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    main()
