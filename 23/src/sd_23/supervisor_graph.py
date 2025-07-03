"""Supervisorパターンのサンプル"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph_supervisor import create_supervisor

from .agents import create_math_agent, create_research_agent

# 環境変数を読み込み
load_dotenv()


def create_supervisor_workflow():
    """Supervisorワークフローを作成（コンパイル前）"""
    # エージェントを作成
    math_agent = create_math_agent()
    research_agent = create_research_agent()

    # Supervisorモデル
    supervisor_model = ChatAnthropic(
        temperature=0, model_name="claude-sonnet-4-20250514"
    )  # type: ignore[call-arg]

    # Supervisorプロンプト
    supervisor_prompt = """あなたはタスクコーディネーターです。

利用可能なエージェント:
- research_expert: 情報収集専門
- math_expert: 計算専門

タスク実行方法:
1. 情報収集が必要な場合、research_expertに委譲
2. 計算が必要な場合、具体的な数値と計算内容をmath_expertに伝える"""

    # Supervisorワークフローを作成
    workflow = create_supervisor(
        agents=[math_agent, research_agent],
        model=supervisor_model,
        prompt=supervisor_prompt,
        add_handoff_messages=True,
        output_mode="full_history",
    )

    return workflow


# LangGraph Studio用のエントリーポイント
graph = create_supervisor_workflow().compile()
