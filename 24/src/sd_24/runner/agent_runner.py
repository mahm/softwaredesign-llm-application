"""エージェント実行ロジック"""

import asyncio
import sys
import argparse
import traceback
from typing import Dict, Any
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from ..main import create_writing_assistant_workflow
from ..display import TerminalUI


class AgentRunner:
    """エージェント実行クラス"""
    
    def __init__(self):
        self.ui = TerminalUI()
        self.debug_mode = False
        
    def parse_command_line_args(self):
        """コマンドライン引数を解析"""
        parser = argparse.ArgumentParser(
            description="LangChain文章執筆支援システム",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用例:
  uv run main.py "LangChainについて教えて"
  uv run main.py "2025年のAI動向をレポートして" --debug
            """
        )
        
        parser.add_argument(
            "query", 
            nargs="?",
            help="エージェントへの指示・質問内容"
        )
        
        parser.add_argument(
            "--debug", 
            action="store_true",
            help="デバッグモードで実行（詳細なイベント情報を表示）"
        )
        
        args = parser.parse_args()
        
        # 引数の設定を適用
        self.debug_mode = args.debug
        
        return args
    
    def get_user_query(self, args) -> str:
        """ユーザークエリを取得"""
        if args.query:
            return args.query
        
        # 引数でクエリが指定されていない場合は対話的に入力を求める
        print("📝 エージェントへの指示を入力してください:")
        print("例: 'LangChainについて調査してレポートを作成して'")
        print("例: '2025年のAI技術動向について詳しく教えて'")
        print()
        
        try:
            query = input("👤 あなた: ").strip()
            if not query:
                print("❌ 指示が入力されませんでした。")
                sys.exit(1)
            return query
        except KeyboardInterrupt:
            print("\n👋 実行をキャンセルしました。")
            sys.exit(0)
    
    def create_execution_config(self) -> Dict[str, Any]:
        """実行設定を作成"""
        return {
            "configurable": {"thread_id": "user_session"},
            "recursion_limit": 100
        }
    
    async def run_agent_execution(
        self,
        app: Any,
        query: str,
        config: Dict[str, Any]
    ):
        """エージェント実行"""
        input_data = {"messages": [HumanMessage(content=query)]}
        
        print("🤖 エージェント実行開始...")
        print(f"📝 指示内容: {query}")
        print("="*60)
        
        try:
            if self.debug_mode:
                await self.ui.run_debug_mode(app, input_data, config, "エージェント実行", query)
            else:
                await self.ui.run_with_task_monitoring(app, input_data, config, "エージェント実行", query)
        
        except KeyboardInterrupt:
            print("\n👋 ユーザーによって実行がキャンセルされました")
            return False
        except Exception as e:
            print(f"\n❌ 実行エラー: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False
        
        return True
    
    async def run_with_query(self, query: str) -> bool:
        """指定されたクエリでエージェントを実行"""
        try:
            # 起動バナー表示
            self.ui.print_startup_banner(self.debug_mode)
            
            # ワークフロー作成
            print("\n🔧 エージェントシステムを初期化中...")
            workflow = create_writing_assistant_workflow()
            app = workflow.compile(checkpointer=InMemorySaver())
            print("✅ 初期化完了")
            
            # 実行設定作成
            config = self.create_execution_config()
            
            # エージェント実行
            success = await self.run_agent_execution(app, query, config)
            
            if success:
                self.ui.print_completion_summary()
            
            return success
            
        except Exception as e:
            self.ui.print_error_summary(e)
            if self.debug_mode:
                print("\n詳細なエラー情報:")
                traceback.print_exc()
            return False
    
    async def run(self) -> bool:
        """メイン実行（コマンドライン引数から）"""
        try:
            # コマンドライン引数解析
            args = self.parse_command_line_args()
            
            # ユーザークエリ取得
            query = self.get_user_query(args)
            
            # エージェント実行
            return await self.run_with_query(query)
            
        except Exception as e:
            print(f"❌ システムエラー: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False