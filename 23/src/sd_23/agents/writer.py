from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from ..utils.memory import memory  # シングルトンインスタンスをインポート

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
            
    return {
        "task_plan": memory.get("task_plan", {}),
        "research": clean_research
    }

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
        "message": "執筆準備完了" if not insufficient_research else "調査が不十分です。追加調査が必要です。",
        "has_research": bool(research_data),
        "has_plan": bool(task_plan)
    }

def create_writer_agent(model: BaseChatModel):
    return create_react_agent(
        model=model,
        tools=[get_all_data, check_writing_readiness],
        name="writer",
        prompt="""
        以下の手順で執筆を行ってください：
        
        1. check_writing_readinessで執筆準備状況を確認
        2. 準備が不十分な場合は、その旨を報告して終了
        3. 準備が整っている場合は、get_all_dataで計画と調査結果を取得
        4. 取得した情報に基づいて、以下の構成で記事を執筆：
           - 導入: 読者の興味を引く内容（調査結果を活用）
           - 本論: 調査結果に基づく詳細な説明（具体的なデータや事例を含む）
           - 結論: 要点のまとめと今後の展望
        
        重要: 調査結果を具体的に引用し、データに基づいた説得力のある記事を作成してください。
        """
    )

# グラフのエクスポート（LangGraph Studio用）
_model = ChatAnthropic(model_name="claude-sonnet-4-20250514")
graph = create_writer_agent(_model)