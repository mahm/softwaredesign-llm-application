import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools.structured import StructuredTool
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool as MCPTool


def load_mcp_config(config_path="mcp_config.json") -> Dict[str, Any]:
    """JSON定義の読み込み"""
    print(f"設定ファイル '{config_path}' を読み込みます...")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_available_servers(config: Dict[str, Any]) -> List[str]:
    """設定ファイルから利用可能なサーバー名のリストを取得します"""
    return list(config.get("mcpServers", {}).keys())


def create_server_params(
    config: Dict[str, Any], server_name: Optional[str] = None
) -> StdioServerParameters:
    """指定されたサーバー名の設定から StdioServerParameters を作成します"""
    available_servers = get_available_servers(config)

    if not available_servers:
        raise ValueError("設定ファイルにMCPサーバーが定義されていません")

    # サーバー名が指定されていない場合は最初のサーバーを使用
    if server_name is None:
        server_name = available_servers[0]
    elif server_name not in available_servers:
        raise ValueError(f"指定されたサーバー '{server_name}' は設定に存在しません")

    # 指定されたサーバーの設定を取得
    server_conf = config["mcpServers"][server_name]
    command = server_conf["command"]
    args = server_conf["args"]

    print(f"サーバー '{server_name}' の設定: command='{command}', args={args}")

    return StdioServerParameters(command=command, args=args, env=dict(os.environ))


def create_all_server_params(
    config: Dict[str, Any],
) -> Dict[str, StdioServerParameters]:
    """設定ファイルに定義されている全てのMCPサーバーのパラメータを作成"""
    servers = {}
    for server_name in get_available_servers(config):
        try:
            servers[server_name] = create_server_params(config, server_name)
        except Exception as e:
            print(f"警告: サーバー '{server_name}' の設定読み込みに失敗しました: {e}")
    return servers


def extract_tool_list(response: Any) -> List[Any]:
    """MCPサーバーのレスポンスからツールリストを抽出します"""
    if hasattr(response, "tools"):
        tool_list = response.tools
        print(f"ツールリストを取得: {len(tool_list)}個")
        return tool_list
    else:
        print("警告: ツールリストを特定できませんでした")
        return []


def extract_tool_info(tool_item: Any) -> Tuple[str, str]:
    """ツールアイテムから名前と説明を抽出します"""
    tool_name = getattr(tool_item, "name", "")
    tool_desc = getattr(tool_item, "description", "")

    if not tool_name:
        print(f"警告: ツール名が見つかりません: {tool_item}")

    return tool_name, tool_desc


async def create_langchain_tool(
    tool_name: str,
    tool_desc: str,
    prefix: str,
    server_name: Optional[str],
    server_params: StdioServerParameters,
    tool_item: MCPTool,
) -> StructuredTool:
    """
    MCPツールをLangChainのStructuredToolとして生成
    """
    # サーバー名をプレフィックスとしてツール名に追加（重複防止）
    full_tool_name = f"{prefix}{tool_name}"
    full_tool_desc = f"[{server_name}] {tool_desc}" if server_name else tool_desc

    try:
        # 非同期の MCP 呼び出し関数を定義
        async def call_mcp_tool_async(**kwargs: Any) -> Any:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=kwargs)
                    return result

        # 同期呼び出し用にラップする
        def tool_func(**kwargs: Any) -> Any:
            return asyncio.run(call_mcp_tool_async(**kwargs))

        # StructuredToolを作成して返す
        return StructuredTool.from_function(
            func=tool_func,
            name=full_tool_name,
            description=full_tool_desc,
            args_schema=tool_item.inputSchema,
        )
    except Exception as e:
        print(f"ツール '{full_tool_name}' の作成中にエラーが発生しました: {str(e)}")
        raise


async def get_mcp_tools(
    session: ClientSession, server_name: Optional[str]
) -> List[Any]:
    """MCPセッションからツールリストを取得します"""
    try:
        print(f"サーバー '{server_name}' からツール一覧を取得しています...")
        response = await session.list_tools()
        return extract_tool_list(response)
    except Exception as e:
        print(f"ツール一覧取得中にエラーが発生しました: {e}")
        return []


async def load_mcp_tools(
    server_params: StdioServerParameters, server_name: Optional[str] = None
) -> List[StructuredTool]:
    """指定したMCPサーバーからツールをロードします"""
    tools: List[StructuredTool] = []
    prefix = f"{server_name}__" if server_name else ""

    print(f"サーバー '{server_name}' に接続しています...")

    try:
        async with stdio_client(server_params) as (read, write):
            print(f"サーバー '{server_name}' への接続が確立されました")

            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"サーバー '{server_name}' のセッション初期化完了")

                # MCPサーバーが提供するツール一覧を取得
                tool_list = await get_mcp_tools(session, server_name)

                # 各ツールを処理
                processed_count = 0
                for tool_item in tool_list:
                    try:
                        # ツール名と説明を取得
                        tool_name, tool_desc = extract_tool_info(tool_item)

                        if not tool_name:
                            continue

                        print(f"ツール処理中: {tool_name}")

                        # StructuredToolを作成
                        lc_tool = await create_langchain_tool(
                            tool_name,
                            tool_desc,
                            prefix,
                            server_name,
                            server_params,
                            tool_item,
                        )
                        tools.append(lc_tool)
                        processed_count += 1
                        print(f"ツール '{tool_name}' が正常に作成されました")
                    except Exception as e:
                        tool_name = getattr(tool_item, "name", str(tool_item))
                        print(f"ツール '{tool_name}' の作成に失敗: {str(e)}")

                print(f"処理したツール数: {processed_count}個")
    except Exception as e:
        print(f"サーバー '{server_name}' との通信に失敗: {e}")

    return tools


async def load_all_mcp_tools(
    config: Optional[Dict[str, Any]] = None,
) -> List[StructuredTool]:
    """全てのMCPサーバーからツールをロードします"""
    if config is None:
        config = load_mcp_config("mcp_config.json")

    all_tools = []
    server_params_dict = create_all_server_params(config)

    # 各サーバーのツールを取得して結合
    for server_name, params in server_params_dict.items():
        print(f"\n--- サーバー '{server_name}' からツールをロード開始 ---")
        server_tools = await load_mcp_tools(params, server_name)
        all_tools.extend(server_tools)
        print(
            f"サーバー '{server_name}' から {len(server_tools)} 個のツールをロードしました"
        )

    return all_tools


if __name__ == "__main__":
    print("MCP Manager を起動しています...")
    try:
        config = load_mcp_config("mcp_config.json")
        # 利用可能なサーバー一覧を表示
        available_servers = get_available_servers(config)
        print(f"利用可能なMCPサーバー: {available_servers}")

        if len(available_servers) > 0:
            # すべてのサーバーからツールをロード
            print("すべてのサーバーからツールをロードしています...")
            all_tools = asyncio.run(load_all_mcp_tools(config))
            print(f"全サーバーからロードされたツール合計: {len(all_tools)}個")

            # ツール一覧を表示
            for tool in all_tools:
                print(f"- {tool.name}: {tool.description}")
        else:
            print(
                "利用可能なサーバーが見つかりません。mcp_config.jsonを確認してください。"
            )
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)
