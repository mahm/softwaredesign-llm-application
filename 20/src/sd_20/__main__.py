import argparse
import uuid

from src.sd_20.agent import graph

if __name__ == "__main__":
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="MCPエージェントと会話する")
    parser.add_argument("query", help="エージェントへの質問")
    args = parser.parse_args()

    # エージェントの実行
    config = {'configurable': {'thread_id': str(uuid.uuid4())}}
    user_input = {"messages": [("human", args.query)]}
    for state in graph.stream(user_input, stream_mode="values", config=config):
        msg = state["messages"][-1]
        msg.pretty_print()
