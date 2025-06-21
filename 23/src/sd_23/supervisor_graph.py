"""Supervisorパターンのシンプルなサンプル"""

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
- math_expert: 数学的な計算や問題解決が得意
- research_expert: 情報収集や調査が得意

ユーザーの要求に基づいて、適切なエージェントに委譲してください。
数学的な計算が必要な場合はmath_expertに、
情報収集が必要な場合はresearch_expertに委譲します。

複数のタスクがある場合は、順番に各エージェントを呼び出してください。"""

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
