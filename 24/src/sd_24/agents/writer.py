from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.memory import memory  # シングルトンインスタンスをインポート
from ..utils.todo_tools import (
    create_get_my_todos_for_agent,
    update_todo_status,
    update_multiple_todo_status,
)  # TODOツールをインポート


@tool
def get_all_data() -> dict:
    """計画と調査結果をすべて取得"""
    research_data = memory.get("research", {})
    # 調査結果から実際の内容だけを抽出
    clean_research = {}
    for topic, data in research_data.items():
        if isinstance(data, dict):
            clean_research[topic] = data.get("findings", data)
        else:
            clean_research[topic] = data

    return {"task_plan": memory.get("task_plan", {}), "research": clean_research}


@tool
def check_writing_readiness() -> dict:
    """執筆の準備が整っているかチェック"""
    research_data = memory.get("research", {})
    task_plan = memory.get("task_plan", {})

    # 不十分な調査があるかチェック
    insufficient_research = any(
        isinstance(data, dict) and data.get("needs_revision", False)
        for data in research_data.values()
    )

    return {
        "ready": not insufficient_research and bool(research_data) and bool(task_plan),
        "message": "執筆準備完了"
        if not insufficient_research
        else "調査が不十分です。追加調査が必要です。",
        "has_research": bool(research_data),
        "has_plan": bool(task_plan),
    }


@tool  # No return_direct - let agent report the result
def save_final_document(content: str) -> str:
    """最終文書をファイルに保存し、ファイルパスを返す"""
    import os
    from datetime import datetime

    # 実行日時を取得
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリを作成 (output/[実行日時]/)
    # 相対パスで指定（実行ディレクトリが24/の場合、24/output/...となる）
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名を生成
    filename = "article.txt"
    filepath = os.path.join(output_dir, filename)

    # ファイルに書き出し
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    # メモリにも保存（既存機能維持）
    memory.set("final_document", content)
    memory.set("final_document_path", filepath)

    return f"最終文書を保存しました: {filepath}"


def create_writer_agent(model: BaseChatModel):
    # writerエージェント専用のget_todos関数
    get_writer_todos = create_get_my_todos_for_agent("writer")

    return create_react_agent(
        model=model,
        tools=[
            get_all_data,
            check_writing_readiness,
            get_writer_todos,
            update_todo_status,
            update_multiple_todo_status,
            save_final_document,
        ],
        name="writer",
        prompt=(
            f"現在日付: {datetime.now().strftime('%Y年%m月%d日')}\n"
            "\n"
            "あなたは文章執筆専門のアシスタントです。\n"
            "\n"
            "重要な制約:\n"
            "- task_decomposerが作成した計画（TODO）に従って執筆を実行\n"
            "- メモリから情報を取得して文章を執筆する\n"
            "- 新たな調査や検索は行わない\n"
            "- 執筆完了時は「執筆完了」とだけ報告する\n"
            "- まとめや総括は一切行わない\n"
            "- 執筆内容の重複出力は厳禁（save_final_documentで保存済み）\n"
            "- 調査結果が不十分な場合は、その旨を報告する\n"
            "\n"
            "手順:\n"
            "1. get_writer_todos\n"
            "2. get_all_data（調査結果一括取得）\n"
            "3. 執筆内容を作成（内部で処理、出力しない）\n"
            "4. update_multiple_todo_status（全COMPLETED）\n"
            "5. save_final_document（執筆内容をファイルに保存）\n"
            "6. 「執筆完了」とだけ報告\n"
            "\n"
            "注意: 執筆内容はツールで保存するため、最終応答で再度出力しない"
        ),
    )


# グラフのエクスポート（LangGraph Studio用）
_model = ChatAnthropic(model_name="claude-sonnet-4-20250514")
graph = create_writer_agent(_model)
