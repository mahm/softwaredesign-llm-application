from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from ..utils.memory import memory  # シングルトンインスタンスをインポート

@tool
def save_research(topic: str, findings: str, needs_revision: bool = False) -> str:
    """調査結果を圧縮して保存し、必要に応じて再調査を提案"""
    # Web検索結果を圧縮
    compressed = memory.compress_research(topic, findings)
    # 既存の調査結果に追加
    research_data = memory.get("research", {})
    research_data[topic] = {
        "findings": compressed,
        "needs_revision": needs_revision,
        "timestamp": str(datetime.now())
    }
    memory.set("research", research_data)
    
    if needs_revision:
        return f"{topic}の調査結果を保存しました。ただし、情報が不十分なため再調査を推奨します。"
    return f"{topic}の調査結果を圧縮して保存しました"

@tool
def get_task_plan() -> dict:
    """保存されたタスク計画を取得"""
    return memory.get("task_plan", {})

@tool
def check_research_sufficiency() -> dict:
    """調査結果の充足度をチェックし、追加調査の必要性を判断"""
    research_data = memory.get("research", {})
    
    insufficient_topics = []
    for topic, data in research_data.items():
        if isinstance(data, dict) and data.get("needs_revision"):
            insufficient_topics.append(topic)
    
    return {
        "sufficient": len(insufficient_topics) == 0,
        "insufficient_topics": insufficient_topics,
        "recommendation": "すべての調査が完了しています" if not insufficient_topics else f"以下のトピックについて追加調査が必要です: {', '.join(insufficient_topics)}"
    }

def create_research_agent(model: BaseChatModel):
    # Tavily検索ツール
    search = TavilySearch(max_results=3)
    
    return create_react_agent(
        model=model,
        tools=[search, save_research, get_task_plan, check_research_sufficiency],
        name="research",
        prompt="""
        以下の手順で調査を行ってください：
        1. get_task_planで計画を取得
        2. 各トピックについてTavilySearchでWeb検索を実行
        3. 検索結果を評価し、情報が不十分な場合はneeds_revision=Trueで保存
        4. すべての調査後、check_research_sufficiencyで充足度を確認
        5. 不十分なトピックがあれば、より詳細なクエリで再検索
        
        重要: 実際にWeb検索を行い、具体的な情報を収集してください。
        """
    )

# グラフのエクスポート（LangGraph Studio用）
_model = ChatAnthropic(model_name="claude-sonnet-4-20250514")
graph = create_research_agent(_model)