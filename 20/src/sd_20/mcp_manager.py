import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.tools.structured import StructuredTool
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field, create_model


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


async def extract_tool_params(tool_item: Any) -> Dict[str, Dict[str, Any]]:
    """ツールアイテムからパラメータ情報を抽出します"""
    params: Dict[str, Dict[str, Any]] = {}

    if not hasattr(tool_item, "inputSchema"):
        return params

    schema = tool_item.inputSchema

    # 辞書として扱う
    if isinstance(schema, dict) and "properties" in schema:
        if isinstance(schema["properties"], dict):
            return schema["properties"]

    # オブジェクトとして扱う
    if hasattr(schema, "properties"):
        props = schema.properties
        if isinstance(props, dict):
            return props
        elif hasattr(props, "__dict__"):
            return props.__dict__

    return params


async def get_required_params(tool_item: Any) -> List[str]:
    """ツールアイテムから必須パラメータのリストを抽出します"""
    if not hasattr(tool_item, "inputSchema"):
        return []

    schema = tool_item.inputSchema

    # requiredフィールドを取得
    required = None
    if hasattr(schema, "required"):
        required = schema.required
    elif isinstance(schema, dict) and "required" in schema:
        required = schema["required"]

    # 必須パラメータをリストとして変換
    if required is None:
        return []
    elif isinstance(required, list):
        return required
    elif isinstance(required, dict):
        return [k for k, v in required.items() if v]
    elif isinstance(required, str):
        try:
            req_data = json.loads(required)
            if isinstance(req_data, list):
                return req_data
            elif isinstance(req_data, dict):
                return [k for k, v in req_data.items() if v]
        except json.JSONDecodeError:
            return [required]

    return []


def create_schema_model(
    tool_name: str, params: Dict[str, Dict[str, Any]], required_params: List[str]
) -> Type[BaseModel]:
    """Pydanticモデルを作成します"""
    # モデル名を設定
    model_name = f"{tool_name.replace('-', '_').capitalize()}Parameters"

    # パラメータがない場合は空のモデルを返す
    if not params:
        return create_model(model_name)

    # フィールド定義を作成
    fields = {}

    for param_name, param_info in params.items():
        # JSONスキーマの型をPythonの型に変換
        python_type: Type[Any] = str

        if "type" in param_info:
            schema_type = param_info["type"]
            if schema_type == "integer":
                python_type = int
            elif schema_type == "number":
                python_type = float
            elif schema_type == "boolean":
                python_type = bool

        # フィールドの説明
        description = param_info.get("description", "")

        # 必須パラメータかどうか
        is_required = param_name in required_params

        # フィールド定義を作成
        if is_required:
            # 必須パラメータ
            fields[param_name] = (python_type, Field(description=description))
        else:
            # オプションパラメータ（デフォルト値あり）
            default_value = param_info.get("default", None)
            fields[param_name] = (
                python_type,
                Field(default=default_value, description=description),
            )

    try:
        # create_modelを使用してモデルを作成
        return create_model(model_name, **fields)  # type: ignore
    except Exception as e:
        print(f"モデル '{model_name}' の作成中にエラーが発生: {str(e)}")
        # 空のモデルにフォールバック
        return create_model(model_name)


async def create_langchain_tool(
    tool_name: str,
    tool_desc: str,
    prefix: str,
    server_name: Optional[str],
    server_params: StdioServerParameters,
    tool_item: Any = None,
) -> StructuredTool:
    """StructuredToolを使ってLangChain Toolオブジェクトを作成します"""
    # サーバー名をプレフィックスとしてツール名に追加（重複防止）
    full_tool_name = f"{prefix}{tool_name}"
    full_tool_desc = f"[{server_name}] {tool_desc}" if server_name else tool_desc

    try:
        # パラメータ情報の抽出
        params = {}
        required_params = []

        if tool_item:
            params = await extract_tool_params(tool_item)
            required_params = await get_required_params(tool_item)

        # Pydanticモデルの作成
        args_schema = create_schema_model(tool_name, params, required_params)

        if params:
            print(f"ツール '{full_tool_name}' のスキーマを作成: {args_schema}")
        else:
            print(f"ツール '{full_tool_name}' に空のスキーマを使用します")

        # 非同期の MCP 呼び出し関数を定義
        async def call_mcp_tool_async(**kwargs: Any) -> Any:
            print(f"ツール '{tool_name}' を呼び出し中、引数: {kwargs}")
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=kwargs)
                    print(f"ツール '{tool_name}' の結果: {result}")
                    return result

        # 同期呼び出し用にラップする
        def tool_func(**kwargs: Any) -> Any:
            return asyncio.run(call_mcp_tool_async(**kwargs))

        # StructuredToolを作成して返す
        return StructuredTool.from_function(
            func=tool_func,
            name=full_tool_name,
            description=full_tool_desc,
            args_schema=args_schema,
        )
    except Exception as e:
        print(f"ツール '{full_tool_name}' の作成中にエラーが発生しました: {str(e)}")
        import traceback

        traceback.print_exc()
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
        import traceback

        traceback.print_exc()

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
