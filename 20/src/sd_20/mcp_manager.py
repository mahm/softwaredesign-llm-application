import asyncio
import json
import os
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import Tool
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def load_mcp_config(config_path="mcp_config.json"):
    """JSON定義の読み込み"""
    print(f"設定ファイル '{config_path}' を読み込みます...")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_available_servers(config: Dict) -> List[str]:
    """設定ファイルから利用可能なサーバー名のリストを取得します"""
    return list(config.get("mcpServers", {}).keys())


def create_server_params(
    config: Dict, server_name: Optional[str] = None
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

    # 環境変数をdictに変換してから渡す
    env_dict = dict(os.environ)
    return StdioServerParameters(command=command, args=args, env=env_dict)


# 複数サーバーのパラメータを作成
def create_all_server_params(config: Dict) -> Dict[str, StdioServerParameters]:
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
    tool_list = []

    # 応答に'tools'属性/キーがある場合、そこからツールリストを取得
    if hasattr(response, "tools"):
        tool_list = response.tools
        print(f"'tools'属性からツールリストを取得: {len(tool_list)}個")
    elif isinstance(response, dict) and "tools" in response:
        tool_list = response["tools"]
        print(f"'tools'キーからツールリストを取得: {len(tool_list)}個")
    else:
        print("警告: ツールリストを特定できませんでした")

    return tool_list


def extract_tool_info(tool_item: Any) -> Tuple[str, str]:
    """ツールアイテムから名前と説明を抽出します"""
    tool_name = ""
    tool_desc = ""

    if hasattr(tool_item, "name"):
        tool_name = tool_item.name
        tool_desc = getattr(tool_item, "description", "")
    elif isinstance(tool_item, dict) and "name" in tool_item:
        tool_name = tool_item["name"]
        tool_desc = tool_item.get("description", "")
    else:
        print(f"無効なツール形式: {tool_item}")

    return tool_name, tool_desc


async def extract_tool_params(tool_item: Any) -> Dict[str, Any]:
    """ツールアイテムからパラメータ情報を抽出します"""
    params = {}

    # MCPツールのinputSchema属性から直接パラメータを抽出
    if hasattr(tool_item, "inputSchema") and isinstance(tool_item.inputSchema, dict):
        schema = tool_item.inputSchema
        if "properties" in schema and isinstance(schema["properties"], dict):
            params = schema["properties"]

    # 辞書形式の場合
    elif isinstance(tool_item, dict) and "inputSchema" in tool_item:
        schema = tool_item["inputSchema"]
        if isinstance(schema, dict) and "properties" in schema:
            if isinstance(schema["properties"], dict):
                params = schema["properties"]

    return params


async def create_langchain_tool(
    tool_name: str,
    tool_desc: str,
    prefix: str,
    server_name: Optional[str],
    server_params: StdioServerParameters,
    tool_item: Any = None,
) -> Tool:
    """LangChain Toolオブジェクトを作成します"""
    # サーバー名をプレフィックスとしてツール名に追加（重複防止）
    full_tool_name = f"{prefix}{tool_name}"

    if server_name:
        full_tool_desc = f"[{server_name}] {tool_desc}"
    else:
        full_tool_desc = tool_desc

    # ツールのパラメータ情報を抽出
    tool_params = {}
    if tool_item:
        tool_params = await extract_tool_params(tool_item)

    print(f"ツールを作成: {full_tool_name}({tool_params})")

    # 非同期の MCP 呼び出し関数を定義
    async def call_mcp_tool(
        original_name=tool_name,
        server_p=server_params,
        **kwargs,
    ):
        async with stdio_client(server_p) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 引数をそのまま渡す
                return await session.call_tool(
                    original_name,
                    arguments=kwargs,
                )

    # 同期呼び出し用にラップする
    def tool_func(*args, **kwargs):
        # デバッグ情報の出力
        print(f"\n==== ツール呼び出しデバッグ情報: {full_tool_name} ====")
        print(f"位置引数: {args}")
        print(f"キーワード引数: {kwargs}")

        # 位置引数がある場合の処理
        if args:
            arg_value = args[0]
            call_kwargs = {}
            print(f"位置引数の型: {type(arg_value)}")
            print(f"位置引数の値: {arg_value}")

            # 辞書の場合はそのまま使用
            if isinstance(arg_value, dict):
                call_kwargs.update(arg_value)
                print(f"辞書として処理: {call_kwargs}")
            # 文字列の場合はクエリパラメータの可能性を考慮
            elif isinstance(arg_value, str):
                # JSONっぽい文字列かどうかチェック（{で始まり}で終わる）
                if arg_value.strip().startswith("{") and arg_value.strip().endswith(
                    "}"
                ):
                    try:
                        # JSON文字列をパースして辞書に変換
                        parsed_json = json.loads(arg_value)
                        if isinstance(parsed_json, dict):
                            call_kwargs.update(parsed_json)
                            print(f"JSON文字列として解析: {parsed_json}")
                        else:
                            first_param = next(iter(tool_params.keys()), "__arg1")
                            call_kwargs[first_param] = arg_value
                            print(
                                f"JSON解析できたが辞書でない: パラメータ '{first_param}' に割り当て"
                            )
                    except json.JSONDecodeError as e:
                        print(f"JSON解析エラー: {e}")
                        first_param = next(iter(tool_params.keys()), "__arg1")
                        call_kwargs[first_param] = arg_value
                        print(f"JSON解析失敗: パラメータ '{first_param}' に割り当て")
                # もし &や= が含まれていたら、クエリパラメータ形式の可能性がある
                elif "&" in arg_value and "=" in arg_value:
                    parsed = parse_query_string(arg_value)
                    if parsed:
                        call_kwargs.update(parsed)
                        print(f"クエリパラメータとして解析: {parsed}")
                    else:
                        # 解析に失敗した場合は単一パラメータとして扱う
                        first_param = next(iter(tool_params.keys()), "__arg1")
                        call_kwargs[first_param] = arg_value
                        print(f"クエリ解析失敗: パラメータ '{first_param}' に割り当て")
                else:
                    # クエリパラメータ形式でない文字列
                    first_param = next(iter(tool_params.keys()), "__arg1")
                    call_kwargs[first_param] = arg_value
                    print(f"単純文字列: パラメータ '{first_param}' に割り当て")
            else:
                # 辞書以外の場合は、最初のパラメータに割り当て
                first_param = next(iter(tool_params.keys()), "__arg1")
                call_kwargs[first_param] = arg_value
                print(f"単一値として処理: パラメータ '{first_param}' に割り当て")

            # 追加のキーワード引数があればマージ
            if kwargs:
                print(f"キーワード引数をマージ: {kwargs}")
            call_kwargs.update(kwargs)

            print(f"最終的な引数: {call_kwargs}")
            print("=" * 50)
            return asyncio.run(call_mcp_tool(**call_kwargs))
        else:
            # 位置引数がない場合は、キーワード引数のみを使用
            print(f"キーワード引数のみで呼び出し: {kwargs}")
            print("=" * 50)
            return asyncio.run(call_mcp_tool(**kwargs))

    return Tool(
        name=full_tool_name,
        func=tool_func,
        description=full_tool_desc,
    )


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
) -> List[Tool]:
    """指定したMCPサーバーからツールをロードします"""
    tools: List[Tool] = []
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

                        # LangChain Toolを作成
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
                    except Exception as e:
                        print(f"ツール '{tool_item}' の作成に失敗: {e}")

                print(f"処理したツール数: {processed_count}個")
    except Exception as e:
        print(f"サーバー '{server_name}' との通信に失敗: {e}")

    return tools


async def load_all_mcp_tools(config: Optional[Dict] = None) -> List[Tool]:
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


def parse_query_string(query_str: str) -> Dict[str, str]:
    """
    URLクエリパラメータ風の文字列（key1=value1&key2=value2形式）を辞書に変換します。

    Args:
        query_str: パース対象の文字列

    Returns:
        パース結果の辞書
    """
    try:
        # URLエンコードされている可能性があるので、デコードする
        return dict(urllib.parse.parse_qsl(query_str))
    except Exception as e:
        print(f"クエリ文字列のパースに失敗: {e}")
        return {}


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
