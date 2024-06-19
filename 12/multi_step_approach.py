import argparse
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from settings import Settings
from tools import search, sufficiency_check, report_writer
from utility import load_prompt, run_streaming_agent

settings = Settings()


def multi_step_agent():
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    tools = [search, sufficiency_check, report_writer]
    return create_react_agent(
        llm, tools=tools, messages_modifier=load_prompt("multi_step_answering_system")
    )


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Process some queries.")
    parser.add_argument("--task", type=str, required=True, help="The query to search")

    # コマンドライン引数を解析
    args = parser.parse_args()

    inputs = {"messages": [("user", args.task)]}
    final_output = run_streaming_agent(multi_step_agent, inputs)
    print("\n\n")
    print("=== final_output ===")
    print(final_output)


if __name__ == "__main__":
    main()
