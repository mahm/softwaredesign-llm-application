from langgraph_supervisor import create_supervisor
from langchain_anthropic import ChatAnthropic
from datetime import datetime
from .agents.task_decomposer import create_task_decomposer
from .agents.research import create_research_agent
from .agents.writer import create_writer_agent


def create_writing_assistant_workflow():
    """文章執筆支援システムのグラフを構築"""

    # モデルの定義
    model = ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=0,
    )

    # エージェントの定義
    agents = [
        create_task_decomposer(),
        create_research_agent(),
        create_writer_agent(),
    ]

    # システムプロンプトの定義
    current_date = datetime.now().strftime('%Y年%m月%d日')
    system_prompt = f"""現在日付: {current_date}

あなたは文章執筆支援システムのコーディネーターです。

役割：
- 各エージェント間の調整と制御フローの管理
- エージェントからの報告に基づく適切な判断と次の行動決定

基本原則：
1. ユーザーからの新規依頼は必ずtask_decomposerで計画を立てる
2. 全エージェントは計画に従って動作する
3. サブエージェントからの報告内容は一切改変しない
4. エージェントが報告した情報（ファイルパス等）はそのまま伝達する

フロー制御：
1. 新規タスク → 必ずtask_decomposerへ
2. 計画完了後 → 計画に従ってresearchへ
3. 調査不足 → researchで追加調査
4. 計画見直しが必要 → task_decomposerへ戻る
5. 十分な情報収集後 → writerへ
6. 執筆完了 → 処理終了

Writer特徴：
- create_react_agentベースの自律的ツール実行
- 小さなツールを組み合わせた段階的な処理
- LLMによる判断（ルールベース処理なし）

最終出力ルール（厳守）：
Writerから「執筆完了」の報告を受けたら：
1. 独自の内容生成は禁止
2. 「執筆結果」「説明のポイント」等の追加は禁止
3. Writerが保存したファイルパスのみを返す
4. 形式: output/[日時]/article.txt（1行のみ）
5. それ以外の文章は一切書かない"""

    # Supervisorワークフロー
    workflow = create_supervisor(
        model=model,
        agents=agents,
        prompt=system_prompt,
    )

    return workflow


# グラフのエクスポート（LangGraph Studio用）
graph = create_writing_assistant_workflow().compile()
