import os
from typing import List

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import AssistantMessage, Message, UserMessage

# 定数定義
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

mcp = FastMCP("sd-19-mcp")


@mcp.tool()
def hello(name: str) -> str:
    """
    与えられた名前に対して挨拶を返します。
    """
    return f"Hello, {name}!"


@mcp.prompt()
def greet_user(user_name: str) -> List[Message]:
    """
    ユーザーに挨拶するための定型プロンプトを返します。
    返値はMCPで規定されたメッセージのリストです。
    """
    return [
        UserMessage(content=f"{user_name}さん、こんにちは。"),
        AssistantMessage(content="どのようにお手伝いできますか？"),
    ]


@mcp.resource("file://readme.md")
def readme() -> str:
    """
    README.md を読み込んで返します。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.normpath(os.path.join(current_dir, "..", "README.md"))

    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README.md not found at: {readme_path}")

    with open(readme_path, "r") as f:
        content = f.read()
        return content


def start_server(
    transport: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:
    """MCPサーバーを起動する

    Args:
        transport (str): 使用するトランスポートモード ('stdio' または 'sse')
        host (str, optional): SSEモード時のホスト名. デフォルトは DEFAULT_HOST
        port (int, optional): SSEモード時のポート番号. デフォルトは DEFAULT_PORT
    """
    try:
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse", host=host, port=port)
        else:
            raise ValueError(f"不正なトランスポートモード: {transport}")
    except Exception as e:
        print(f"サーバー起動エラー: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCPサーバーの起動モードを指定")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="使用するトランスポートモード (stdio または sse)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="ホスト名",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="ポート番号",
    )
    args = parser.parse_args()
    start_server(
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
