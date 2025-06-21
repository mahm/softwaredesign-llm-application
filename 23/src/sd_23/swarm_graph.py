"""Swarmパターンのシンプルなサンプル"""

from dotenv import load_dotenv
from langgraph_swarm import create_swarm

from .agents import create_faq_agent, create_tech_agent

# 環境変数を読み込み
load_dotenv()


def create_swarm_workflow():
    """Swarmワークフローを作成（コンパイル前）"""
    # エージェントを作成
    faq_agent = create_faq_agent()
    tech_agent = create_tech_agent()

    # Swarmワークフローを作成
    workflow = create_swarm(
        agents=[faq_agent, tech_agent],
        default_active_agent="faq_support",  # 最初はFAQサポートが対応
    )

    return workflow


# LangGraph Studio用のエントリーポイント
graph = create_swarm_workflow().compile()
