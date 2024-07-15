from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import report_writer, search, sufficiency_check
from utility import load_prompt, run_streaming_agent


def create_multi_step_agent(llm: ChatOpenAI):
    tools = [search, sufficiency_check, report_writer]
    return create_react_agent(
        llm, tools=tools, messages_modifier=load_prompt("multi_step_answering_system")
    )


def main():
    import argparse

    from settings import Settings

    settings = Settings()

    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Process some queries.")
    parser.add_argument("--task", type=str, required=True, help="The query to search")

    # コマンドライン引数を解析
    args = parser.parse_args()

    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    inputs = {"messages": [("user", args.task)]}
    agent = create_multi_step_agent(llm=llm)
    final_output = run_streaming_agent(agent, inputs)
    print("\n\n")
    print("=== final_output ===")
    print(final_output)


if __name__ == "__main__":
    main()
