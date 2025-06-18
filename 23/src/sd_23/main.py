from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from .agents.task_decomposer import create_task_decomposer
from .agents.research import create_research_agent
from .agents.writer import create_writer_agent

def create_writing_assistant():
    """文章執筆支援システムのグラフを構築"""
    
    # モデル設定
    model = ChatAnthropic(model_name="claude-sonnet-4-20250514", temperature=0.7)
    
    # エージェントの作成（シングルトンメモリを内部で使用）
    task_decomposer = create_task_decomposer(model)
    research_agent = create_research_agent(model)
    writer_agent = create_writer_agent(model)
    
    # Supervisorプロンプト
    supervisor_prompt = """
    文章執筆支援システムのコーディネーターです。
    
    基本フロー:
    1. task_decomposer: 要望を分析し計画作成
    2. research: 必要な情報を収集
    3. writer: 文章を執筆
    
    重要な調整ルール:
    - researchが「情報が不十分」と報告した場合、task_decomposerに戻って計画を修正
    - writerが「調査が不十分」と報告した場合、researchに追加調査を依頼
    - 各エージェントは最大3回まで再試行可能
    - エージェント間の対話を促し、品質向上を図る
    
    各エージェントの出力を詳細に確認し、必要に応じて別のエージェントに再依頼してください。
    """
    
    # Supervisorワークフロー
    workflow = create_supervisor(
        agents=[task_decomposer, research_agent, writer_agent],
        model=model,
        prompt=supervisor_prompt
    )
    
    # インメモリチェックポインター
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)

# グラフのエクスポート（LangGraph Studio用）
graph = create_writing_assistant()