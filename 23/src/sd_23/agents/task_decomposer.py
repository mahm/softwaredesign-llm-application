from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from ..utils.memory import memory  # シングルトンインスタンスをインポート

@tool
def save_task_plan(plan: dict) -> str:
    """タスク計画を保存"""
    # 既存の計画があれば履歴として保存
    existing_plan = memory.get("task_plan")
    if existing_plan:
        history = memory.get("task_plan_history", [])
        if isinstance(history, list):
            history.append(existing_plan)
        else:
            history = [existing_plan]
        memory.set("task_plan_history", history)
    
    memory.set("task_plan", plan)
    return "タスク計画を保存しました"

@tool
def get_research_feedback() -> dict:
    """調査結果のフィードバックを取得して計画修正の必要性を判断"""
    research_data = memory.get("research", {})
    feedback = {
        "has_feedback": bool(research_data),
        "topics_with_issues": [],
        "suggestions": []
    }
    
    for topic, data in research_data.items():
        if isinstance(data, dict) and data.get("needs_revision"):
            feedback["topics_with_issues"].append(topic)
            feedback["suggestions"].append(f"{topic}についてより具体的な調査項目を追加")
    
    return feedback

def create_task_decomposer(model: BaseChatModel):
    return create_react_agent(
        model=model,
        tools=[save_task_plan, get_research_feedback],
        name="task_decomposer",
        prompt="""
        以下の手順でタスクを分解してください：
        
        1. まずget_research_feedbackで既存の調査結果のフィードバックを確認
        2. フィードバックがある場合は、それを考慮して計画を修正
        3. ユーザーの執筆要望を分析し、以下の構造で計画：
           - 調査が必要なトピック（2-3個、具体的に）
           - 文章構成（導入・本論・結論）
           - 各セクションの概要
        4. save_task_planツールで計画を保存
        
        重要: 調査フィードバックに基づいて、より詳細で実現可能な計画を作成してください。
        """
    )

# グラフのエクスポート（LangGraph Studio用）
_model = ChatAnthropic(model_name="claude-sonnet-4-20250514")
graph = create_task_decomposer(_model)