"""LangChain文章執筆支援システム - Claude Code風UI統合版"""

import asyncio
import sys
from dotenv import load_dotenv
from src.sd_24.runner import AgentRunner
import warnings

# PythonのSyntaxWarningを無視
warnings.filterwarnings("ignore", category=SyntaxWarning)

# 環境変数を読み込み
load_dotenv()


async def main():
    """メイン関数"""
    runner = AgentRunner()
    success = await runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())