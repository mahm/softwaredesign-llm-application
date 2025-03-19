import argparse

from src.sd_20.agent import graph


def get_user_input():
    """コマンドラインからユーザー入力を取得する"""
    parser = argparse.ArgumentParser(description="MCPエージェントと会話する")
    parser.add_argument("query", nargs="?", help="エージェントへの質問", default=None)
    args = parser.parse_args()

    # コマンドライン引数が指定されていない場合は対話的に入力を受け取る
    if args.query is None:
        print("エージェントに質問してください (Ctrl+Dで終了):")
        query = input("> ")
    else:
        query = args.query

    return {"messages": [("user", query)]}


if __name__ == "__main__":
    # ユーザーの入力をコマンドラインから取得
    user_input = get_user_input()

    # エージェントの実行
    for state in graph.stream(user_input, stream_mode="values"):
        msg = state["messages"][-1]
        msg.pretty_print()
