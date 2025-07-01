"""技術サポートに特化したエージェント"""

import random
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool
from langgraph.graph.graph import CompiledGraph


@tool
def check_system_status() -> str:
    """システムステータスをチェックします（モック実装）"""
    return "システムステータス: すべてのサービスが正常に動作しています。"


@tool
def read_manual(error_code: str) -> str:
    """エラーコードのマニュアルを読み取ります（モック実装）"""
    if error_code == "500":
        return """エラーコード500 (Internal Server Error)

【原因】
1. サーバー側のプログラムエラー
2. データベース接続エラー
3. メモリ不足やリソース枯渇

【診断方法】
- サーバーログの確認
- 最近のデプロイメントの確認
- リソース使用状況の確認

【解決方法】
1. ブラウザのキャッシュをクリアして再読み込み
2. 5分待ってから再度アクセス
3. 別のブラウザやデバイスでアクセス
4. それでも解決しない場合はサポートチケット作成"""
    return f"エラーコード{error_code}のマニュアル情報は見つかりませんでした。"


def create_tech_agent() -> CompiledGraph:
    """技術サポートエージェントを作成"""
    model = ChatAnthropic(temperature=0, model_name="claude-sonnet-4-20250514")

    # FAQサポートへのハンドオフツール
    faq_handoff = create_handoff_tool(
        agent_name="faq_support",
        description="一般的なFAQサポートが必要な場合はFAQサポートに転送",
    )

    tools = [check_system_status, read_manual, faq_handoff]

    prompt = """あなたの名前は「tech_support」です。技術サポートエージェントとして動作しています。

役割：
- エラーコードの詳細な診断と解決策の提供
- システム状態の確認とトラブルシューティング
- 必要に応じてサポートチケットの作成

エラーコードの診断手順：
1. check_system_statusツールでシステム状態を確認
2. read_manualツールでエラーコードの詳細を確認
3. 両方の結果を基に具体的な解決策を提供

重要：
- あなたは「tech_support」という名前のエージェントです
- FAQエージェントから転送された技術的な問題に対応します
- 必ずツールを使用して具体的な診断を行ってください
- プロダクトに関する一般的なFAQは、faq_supportへ転送してください"""

    agent = create_react_agent(
        model=model, tools=tools, name="tech_support", prompt=prompt
    )

    return agent
