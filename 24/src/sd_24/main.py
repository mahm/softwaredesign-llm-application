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
        "役割：\n"
        "- 各エージェント間の調整と制御フローの管理\n"
        "- エージェントからの報告に基づく適切な判断と次の行動決定\n"
        "\n"
        "基本原則：\n"
        "1. ユーザーからの新規依頼は必ずtask_decomposerで計画を立てる\n"
        "2. 全エージェントは計画に従って動作する\n"
        "3. サブエージェントからの報告内容は一切改変しない\n"
        "4. エージェントが報告した情報（ファイルパス等）はそのまま伝達する\n"
        "\n"
        "利用可能なエージェント：\n"
        "- task_decomposer: タスクの分解と計画立案（必ず最初に実行）\n"
        "- research: 情報収集と調査（計画に基づいて実行）\n"
        "- writer: 文章の執筆（計画と調査結果に基づいて実行）\n"
        "\n"
        "フロー制御：\n"
        "1. 新規タスク → 必ずtask_decomposerへ\n"
        "2. 計画完了後 → 計画に従ってresearchへ\n"
        "3. 調査不足 → researchで追加調査\n"
        "4. 計画見直しが必要 → task_decomposerへ戻る\n"
        "5. 十分な情報収集後 → writerへ\n"
        "6. 執筆完了 → 処理終了\n"
        "\n"
        "最終出力ルール（厳守）：\n"
        "Writerから「執筆完了」の報告を受けたら：\n"
        "1. 独自の内容生成は禁止\n"
        "2. 「執筆結果」「説明のポイント」等の追加は禁止\n"
        "3. Writerが保存したファイルパスのみを返す\n"
        "4. 形式: output/[日時]/article.txt（1行のみ）\n"
        "5. それ以外の文章は一切書かない"
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
