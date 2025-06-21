from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from .agents.task_decomposer import create_task_decomposer
from .agents.research import create_research_agent
from .agents.writer import create_writer_agent


def create_writing_assistant_workflow():
    """文章執筆支援システムのグラフを構築"""

    # モデル設定
    model = ChatAnthropic(
        model_name="claude-sonnet-4-20250514", temperature=0.7)

    # エージェントの作成（シングルトンメモリを内部で使用）
    task_decomposer = create_task_decomposer(model)
    research_agent = create_research_agent(model)
    writer_agent = create_writer_agent(model)

    # Supervisorプロンプト
    supervisor_prompt = (
        f"現在日付: {datetime.now().strftime('%Y年%m月%d日')}\n"
        "\n"
        "文章執筆支援システムのコーディネーターです。\n"
        "\n"
        "フロー:\n"
        "1. task_decomposer: 「TODOを作成しました」と報告したらresearchに移行\n"
        "2. research: 「調査完了」と報告したらwriterに移行\n"
        "3. writer: 「執筆完了」と報告したら終了\n"
        "\n"
        "最終出力:\n"
        "writerが「最終文書を保存しました: ファイルパス」と報告したら、\n"
        "そのファイルパスのみをユーザーに返してください。\n"
        "他の詳細な経過は含めず、ファイルパスだけを簡潔に伝える。"
    )

    # Supervisorワークフロー
    workflow = create_supervisor(
        agents=[task_decomposer, research_agent, writer_agent],
        model=model,
        prompt=supervisor_prompt,
    )

    return workflow


# グラフのエクスポート（LangGraph Studio用）
graph = create_writing_assistant_workflow().compile()
