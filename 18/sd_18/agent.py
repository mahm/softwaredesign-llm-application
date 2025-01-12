from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


def create_tavily_tool() -> BaseTool:
    """Tavily検索ツールを生成します。

    Returns:
        BaseTool: 最大5件の検索結果を返すTavily検索ツール
    """
    return TavilySearchResults(max_results=5)


def create_python_repl_tool() -> BaseTool:
    """Python REPL実行ツールを生成します。

    このツールは、与えられたPythonコードを安全に実行し、
    実行結果を整形して返します。エラーが発生した場合は
    エラーメッセージを適切に整形して返します。

    Returns:
        BaseTool: Pythonコードを実行するためのREPLツール
    """
    repl = PythonREPL()

    @tool
    def python_repl_tool(code: str) -> str:
        """Pythonコードを実行するためのツール

        Args:
            code: 実行するPythonコードの文字列

        Returns:
            str: 実行結果またはエラーメッセージ
                 成功時: コードと実行結果を含むマークダウン形式の文字列
                 失敗時: エラーの詳細を含む文字列
        """
        try:
            result = repl.run(code)
            return f"実行完了:\n```python\n{code}\n```\n実行結果: {result}"
        except BaseException as e:
            return f"実行に失敗しました。エラー: {repr(e)}"

    return python_repl_tool


@dataclass
class AgentConfig:
    """エージェントの設定を保持するデータクラス

    Attributes:
        prompt (str): システムプロンプト。エージェントの役割や制約を定義します。
        tools (List[BaseTool]): エージェントが使用できるツールのリスト。
        name (str): エージェントの識別名。メッセージのやり取りで使用されます。
    """

    prompt: str
    tools: List[BaseTool]
    name: str


class Agent(ABC):
    """エージェント基底クラス

    このクラスは、LangChainとAnthropicのClaudeモデルを使用して
    特定のタスクを実行するエージェントの基本機能を提供します。

    Attributes:
        config (AgentConfig): エージェントの設定
        llm (ChatAnthropic): Claude LLM
        agent (Any): ReActエージェントのインスタンス
    """

    def __init__(self, config: AgentConfig) -> None:
        """エージェントを初期化します。

        Args:
            config: エージェントの設定を含むAgentConfigインスタンス
        """
        self.config = config
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
        self.agent = create_react_agent(
            self.llm,
            tools=self.config.tools,
            state_modifier=self._make_system_prompt(),
        )

    def _make_system_prompt(self) -> str:
        """システムプロンプトを生成します。

        基本的なプロンプトとエージェント固有のプロンプトを組み合わせて
        完全なシステムプロンプトを生成します。

        Returns:
            str: 完全なシステムプロンプト
        """
        base_prompt = (
            "あなたは他のAIアシスタントと協力して作業を行うアシスタントです。\n"
            "与えられたツールを使用して、質問に対する回答を進めてください。\n"
            "完全な回答ができない場合でも問題ありません。他のアシスタントが続きを担当します。\n"
            "できる範囲で作業を進めてください。\n"
            "あなたまたは他のアシスタントが最終的な回答や成果物を得た場合は、\n"
            "回答の冒頭に「FINAL ANSWER」と記載して、作業完了を示してください。\n\n"
        )
        return base_prompt + self.config.prompt

    def run(self, state: MessagesState, next_node: str) -> Command:
        """エージェントを実行し、次のノードを決定します。

        Args:
            state: 現在のメッセージ状態
            next_node: 次に実行するノードの名前

        Returns:
            Command: 次のノードとメッセージの更新情報を含むコマンド
        """
        result = self.agent.invoke(state)
        goto = END if "FINAL ANSWER" in result["messages"][-1].content else next_node
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name=self.config.name
        )
        return Command(update={"messages": result["messages"]}, goto=goto)


class ResearchAgent(Agent):
    """リサーチを行うエージェント

    Tavilyの検索APIを使用して情報を収集し、
    データをCSV形式でまとめるようにCodeGeneratorに依頼します。
    """

    def __init__(self) -> None:
        """リサーチエージェントを初期化します。"""
        prompt = (
            "あなたはデータリサーチの専門家です。\n"
            "コードジェネレータと協力して作業を行います。\n\n"
            "### あなたの役割\n"
            "1. 要求された情報を検索して収集\n"
            "2. 収集したデータはCSV形式でまとめるようにコードジェネレータに依頼\n"
            "3. データの形式や期間が不明な場合の基準：\n"
            "   - 数値データは可能な限り詳細な粒度で収集\n"
            "   - 期間未指定の場合は直近1年間\n"
            "   - 時系列データは日次または月次を優先\n\n"
            "### 注意事項\n"
            "- 信頼できる情報源を選択\n"
            "- データの出典を記録\n"
            "- 数値データの単位を明記"
        )
        super().__init__(
            AgentConfig(
                prompt=prompt,
                tools=[create_tavily_tool()],
                name="researcher",
            )
        )


class CodeGenerator(Agent):
    """コード生成を行うエージェント

    ResearchAgentから受け取ったデータを基に、
    Pythonコードを生成してデータの処理と可視化を行います。
    """

    def __init__(self, timestamp: str) -> None:
        """コードジェネレータを初期化します。

        Args:
            timestamp: 出力ファイルの保存に使用するタイムスタンプ
        """
        self._setup_visualization_env()
        prompt = self._create_prompt(timestamp)
        super().__init__(
            AgentConfig(
                prompt=prompt,
                tools=[create_python_repl_tool()],
                name="code_generator",
            )
        )

    def _create_prompt(self, timestamp: str) -> str:
        """プロンプトを生成します。

        タイムスタンプを含む出力パスと、
        チャート生成およびデータセット作成の詳細な手順を含むプロンプトを生成します。

        Args:
            timestamp: 出力ファイルの保存に使用するタイムスタンプ

        Returns:
            str: 生成されたプロンプト
        """
        return (
            "あなたはPythonコードを生成する専門家です。\n"
            "リサーチャーと協力してデータの処理と可視化を行います。\n\n"
            "### チャート生成\n"
            f"1. 保存先: 'output/{timestamp}/charts/'\n"
            "2. ファイル名は内容が分かる具体的な名前\n"
            "3. plt.savefig()で保存してから表示\n"
            "4. 日本語フォント設定:\n"
            "   ```python\n"
            "   import matplotlib.pyplot as plt\n"
            "   plt.rcParams['font.family'] = 'IPAGothic'\n"
            "   ```\n\n"
            "### データセット作成\n"
            f"1. 保存先: 'output/{timestamp}/data/'\n"
            "2. ファイル名は内容が分かる具体的な名前\n"
            "3. CSV形式、UTF-8エンコーディング\n"
            "4. 例:\n"
            "   ```python\n"
            "   import pandas as pd\n"
            f"   df.to_csv('output/{timestamp}/data/dataset.csv', index=False, encoding='utf-8')\n"
            "   ```\n\n"
            "### グラフ作成基準\n"
            "1. タイトル・軸ラベルは日本語\n"
            "2. 凡例は必要時のみ\n"
            "3. グリッド線は適宜追加\n"
            "4. 時系列データは折れ線グラフ"
        )

    @staticmethod
    def _setup_visualization_env() -> None:
        """可視化環境をセットアップします。

        以下の設定を行います：
        1. matplotlibのインポートと初期設定
        2. 日本語フォントの設定
           - japanize_matplotlibパッケージを優先使用
           - 利用できない場合は、システムフォント（IPAGothicまたはNoto Sans CJK JP）を使用
        """
        import matplotlib.pyplot as plt

        try:
            import japanize_matplotlib
        except ImportError:
            import matplotlib.font_manager as fm

            plt.rcParams["font.family"] = "IPAGothic"
            if "IPAGothic" not in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams["font.family"] = "Noto Sans CJK JP"


class WorkflowGraph:
    """ワークフローを管理するクラス

    ResearchAgentとCodeGeneratorを組み合わせて、
    データの収集から可視化までの一連の処理を管理します。

    Attributes:
        timestamp (str): 実行時のタイムスタンプ
        research_agent (ResearchAgent): リサーチを行うエージェント
        code_generator (CodeGenerator): コード生成を行うエージェント
        workflow (Any): コンパイル済みのワークフローグラフ
    """

    def __init__(self) -> None:
        """ワークフローを初期化します。"""
        self.timestamp = self._get_timestamp()
        self._ensure_output_dirs()
        self.research_agent = ResearchAgent()
        self.code_generator = CodeGenerator(self.timestamp)
        self.workflow = self._create_workflow()

    @staticmethod
    def _get_timestamp() -> str:
        """実行時のタイムスタンプを取得します。

        Returns:
            str: YYYYMMDD_HHMMSS形式のタイムスタンプ
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _ensure_output_dirs(self) -> None:
        """出力ディレクトリを作成します。

        タイムスタンプ付きのディレクトリ構造を作成します：
        output/
        └── {timestamp}/
            ├── charts/  # チャートファイル用
            └── data/    # データセット用
        """
        import os

        os.makedirs(f"output/{self.timestamp}/charts", exist_ok=True)
        os.makedirs(f"output/{self.timestamp}/data", exist_ok=True)

    def _create_workflow(self) -> StateGraph:
        """ワークフローを生成します。

        ResearchAgentとCodeGeneratorを接続し、
        直列に実行されるワークフローを構築します。

        Returns:
            StateGraph: コンパイル済みのワークフローグラフ
        """
        workflow = StateGraph(MessagesState)
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("code_generator", self._code_node)
        workflow.add_edge(START, "researcher")
        return workflow.compile()

    def _research_node(self, state: MessagesState) -> Command:
        """リサーチノードを実行します。

        Args:
            state: 現在のメッセージ状態

        Returns:
            Command: 次のノードとメッセージの更新情報を含むコマンド
        """
        return self.research_agent.run(state, "code_generator")

    def _code_node(self, state: MessagesState) -> Command:
        """コード生成ノードを実行します。

        Args:
            state: 現在のメッセージ状態

        Returns:
            Command: 次のノードとメッセージの更新情報を含むコマンド
        """
        return self.code_generator.run(state, "researcher")

    def run(self, message: str) -> None:
        """ワークフローを実行します。

        Args:
            message: ユーザーからの初期メッセージ。
                    データの収集と可視化に関する要求を含みます。
        """
        events = self.workflow.stream(
            {"messages": [("user", message)]},
            {"recursion_limit": 150},  # 最大ステップ数
        )
        for event in events:
            print(event)
            print("----")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    workflow = WorkflowGraph()
    workflow.run(
        "2024年の1年間の日経平均推移をCSVファイルとチャートにしてください。"
        "チャートとCSVファイルを作成したら終了してください。"
    )
