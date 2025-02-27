from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger

logger = get_logger(__name__)
configure_logging()

server_params = StdioServerParameters(
    command="python",
    args=["sd_19/server.py"],
    env=None,
)


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 接続を初期化
            await session.initialize()

            # 利用可能なプロンプト一覧を取得
            prompts = await session.list_prompts()
            logger.info("=== prompts ===")
            logger.info(prompts)
            logger.info("\n")

            # プロンプトを取得
            prompt = await session.get_prompt(
                "greet_user", arguments={"user_name": "sd-19"}
            )
            logger.info("=== prompt ===")
            logger.info(prompt)
            logger.info("\n")

            # 利用可能なリソース一覧を取得
            resources = await session.list_resources()
            logger.info("=== resources ===")
            logger.info(resources)
            logger.info("\n")

            # リソースを読み込む
            resource = await session.read_resource("file://readme.md")
            logger.info("=== resource ===")
            logger.info(resource)
            logger.info("\n")

            # 利用可能なツール一覧を取得
            tools = await session.list_tools()
            logger.info("=== tools ===")
            logger.info(tools)
            logger.info("\n")

            # ツールを呼び出す
            result = await session.call_tool("hello", arguments={"name": "sd-19"})
            logger.info("=== tool result ===")
            logger.info(result)
            logger.info("\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
