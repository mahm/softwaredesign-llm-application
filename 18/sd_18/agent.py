from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


def create_tavily_tool() -> BaseTool:
    """Tavily検索ツールを生成"""
    return TavilySearchResults(max_results=5)


def create_python_repl_tool() -> BaseTool:
    """Python REPL実行ツールを生成"""
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


class FileReader:
    """ファイル読み込みツールの提供を行う"""

    def __init__(self, timestamp: str) -> None:
        self.timestamp = timestamp

    def create_tool(self) -> BaseTool:
        """ファイル読み込みツールを生成"""
        timestamp = self.timestamp

        @tool
        def file_reader(file_path: str) -> str:
            """ファイルの内容を読み込むツール

            Args:
                file_path: 読み込むファイルのパス
                         - 完全なパス（output/{timestamp}/data/file.csv）
                         - 相対パス（data/file.csv）のいずれかを指定する

            Returns:
                str: ファイルの内容
                     CSVの場合は先頭10行のデータを返す
                     画像の場合はファイルの基本情報を返す
            """
            import os
            from pathlib import Path

            def read_file(path: Path) -> Optional[str]:
                """ファイルを読み込み、内容を返す

                Args:
                    path: 読み込むファイルのパス

                Returns:
                    Optional[str]: ファイルの内容。ファイルが存在しない場合はNoneを返す
                """
                if not path.exists():
                    return None

                try:
                    if path.suffix.lower() == ".csv":
                        import pandas as pd

                        df = pd.read_csv(path, encoding="utf-8")
                        preview = df.head(10).to_string()
                        return (
                            f"CSVファイル情報:\n"
                            f"- パス: {path}\n"
                            f"- 行数: {len(df)}\n"
                            f"- カラム: {', '.join(df.columns)}\n"
                            f"- データプレビュー（先頭10行）:\n{preview}"
                        )

                    elif path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        from PIL import Image

                        img = Image.open(path)
                        return (
                            f"画像ファイル情報:\n"
                            f"- パス: {path}\n"
                            f"- サイズ: {img.size}\n"
                            f"- フォーマット: {img.format}\n"
                            f"- モード: {img.mode}"
                        )

                    else:
                        return f"エラー: サポートされていないファイル形式です: {path.suffix}"

                except Exception as e:
                    return f"ファイル読み込みエラー: {str(e)}"

            try:
                # 与えられたパスをPathオブジェクトに変換
                file_path = Path(file_path)

                # パターン1: 完全なパス（output/{timestamp}/から始まる）
                if str(file_path).startswith(f"output/{timestamp}/"):
                    result = read_file(file_path)
                    if result:
                        return result

                # パターン2: 相対パス（data/やcharts/から始まる）
                timestamped_path = Path(f"output/{timestamp}/{file_path}")
                result = read_file(timestamped_path)
                if result:
                    return result

                # どちらのパスでもファイルが見つからない場合
                return (
                    f"エラー: ファイルが見つかりません。\n"
                    f"試行したパス:\n"
                    f"1. {file_path}\n"
                    f"2. {timestamped_path}"
                )

            except Exception as e:
                return f"ファイル読み込みエラー: {str(e)}"

        return file_reader


@dataclass
class AgentConfig:
    """エージェントの設定を保持するデータクラス

    Attributes:
        prompt (str): システムプロンプト。エージェントの役割や制約を定義する。
        tools (List[BaseTool]): エージェントが使用できるツールのリスト。
        name (str): エージェントの識別名。メッセージのやり取りで使用する。
    """

    prompt: str
    tools: List[BaseTool]
    name: str


class Agent(ABC):
    """エージェント

    特定のタスクを実行するエージェントの基本機能を提供する。

    Attributes:
        config (AgentConfig): エージェントの設定
        llm (ChatAnthropic): Claude LLM
        agent (Any): ReActエージェントのインスタンス
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
        self.agent = create_react_agent(
            self.llm,
            tools=self.config.tools,
            state_modifier=self._make_system_prompt(),
        )

    def _make_system_prompt(self) -> str:
        """システムプロンプトを生成"""
        base_prompt = (
            "あなたは他のAIアシスタントと協力して作業を行うアシスタントです。\n"
            "与えられたツールを使用して、質問に対する回答を進めること。\n"
            "完全な回答ができない場合でも問題ありません。他のアシスタントが続きを担当します。\n"
            "できる範囲で作業を進めること。\n"
        )
        return base_prompt + self.config.prompt

    def run(self, state: MessagesState, next_node: str) -> Command:
        """エージェントを実行"""
        result = self.agent.invoke(state)
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name=self.config.name
        )
        return Command(update={"messages": result["messages"]}, goto=next_node)


def _read_prompt(file_name: str, **kwargs) -> str:
    """プロンプトファイルを読み込む"""
    with open(f"prompts/{file_name}", "r", encoding="utf-8") as f:
        prompt = f.read()

    # 変数の置換
    if kwargs:
        prompt = prompt.format(**kwargs)

    return prompt


class ResearchAgent(Agent):
    """データ検索と収集を行うエージェント

    Tavily Search APIを使用して情報を収集し、結果を構造化する。
    """

    def __init__(self) -> None:
        prompt = _read_prompt("researcher.prompt")
        super().__init__(
            AgentConfig(
                prompt=prompt,
                tools=[create_tavily_tool()],
                name="researcher",
            )
        )


class CodeGenerator(Agent):
    """データセットとチャートを生成するエージェント

    収集されたデータを基に、CSVデータセットとチャートを生成する。
    生成されたファイルはタイムスタンプ付きディレクトリに保存される。
    """

    def __init__(self, timestamp: str) -> None:
        self._setup_visualization_env()
        prompt = _read_prompt("code_generator.prompt", timestamp=timestamp)
        super().__init__(
            AgentConfig(
                prompt=prompt,
                tools=[create_python_repl_tool()],
                name="code_generator",
            )
        )

    @staticmethod
    def _setup_visualization_env() -> None:
        """可視化環境をセットアップ

        以下の設定を行う：
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


class ReflectionAgent(Agent):
    """データの検証と評価を行うエージェント

    収集されたデータと生成された成果物の品質を検証する。
    検証結果に基づいて次のステップを決定する。
    """

    def __init__(self, file_reader: FileReader) -> None:
        prompt = _read_prompt("reflection.prompt")
        super().__init__(
            AgentConfig(
                prompt=prompt,
                tools=[
                    create_tavily_tool(),
                    file_reader.create_tool(),
                ],
                name="reflection",
            )
        )

    def run(self, state: MessagesState, next_node: str) -> Command:
        """エージェントを実行し、検証結果に基づいて次のノードを決定する

        Args:
            state: 現在のメッセージ状態
            next_node: デフォルトの次のノード

        Returns:
            Command: 次のノードとメッセージの更新情報を含むコマンド
                - 検証OKの場合: code_generatorに進む
                - 検証NGの場合: next_nodeで指定するノードに差し戻す
                - 最終確認OKの場合: ENDに進む
        """
        result = self.agent.invoke(state)
        content = result["messages"][-1].content

        # 次のノードの指定を検出
        import re

        next_node_match = re.search(
            r"次のノード:\s*(researcher|code_generator|end)", content
        )
        if next_node_match:
            goto = next_node_match.group(1)
            if goto == "end":
                goto = END
        else:
            # 次のノードが不明確な場合は改善を要求
            goto = next_node
            content += (
                "\n\n次のノードが不明確です。正しい形式で次のノードを指定してください。"
            )

        result["messages"][-1] = HumanMessage(content=content, name=self.config.name)
        return Command(update={"messages": result["messages"]}, goto=goto)


class WorkflowGraph:
    def __init__(self) -> None:
        self.timestamp = self._get_timestamp()
        self._ensure_output_dirs()
        self.research_agent = ResearchAgent()
        self.code_generator = CodeGenerator(self.timestamp)
        self.file_reader = FileReader(self.timestamp)
        self.reflection_agent = ReflectionAgent(self.file_reader)
        self.workflow = self._create_workflow()

    @staticmethod
    def _get_timestamp() -> str:
        """実行時のタイムスタンプを取得する

        Returns:
            str: YYYYMMDD_HHMMSS形式のタイムスタンプ
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _ensure_output_dirs(self) -> None:
        """出力ディレクトリを作成する

        タイムスタンプ付きのディレクトリ構造を作成する：
        - output/{timestamp}/data/: データセット用
        - output/{timestamp}/charts/: チャート用
        """
        import os

        os.makedirs(f"output/{self.timestamp}/charts", exist_ok=True)
        os.makedirs(f"output/{self.timestamp}/data", exist_ok=True)

    def _create_workflow(self) -> CompiledStateGraph:
        """ワークフローを生成する"""
        workflow = StateGraph(MessagesState)

        # ノードの追加
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("code_generator", self._code_node)

        # 開始ノードの設定
        workflow.set_entry_point("researcher")

        return workflow.compile()

    def _research_node(
        self, state: MessagesState
    ) -> Command[Literal["reflection", END]]:
        """リサーチノードの実行"""
        return self.research_agent.run(state, "reflection")

    def _reflection_node(
        self, state: MessagesState
    ) -> Command[Literal["researcher", "code_generator", END]]:
        """リフレクションノードの実行"""
        return self.reflection_agent.run(state, "researcher")

    def _code_node(self, state: MessagesState) -> Command[Literal[END]]:
        """コード生成ノードの実行"""
        return self.code_generator.run(state, "reflection")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    workflow = WorkflowGraph()
    message = (
        "2024年の1年間の日経平均推移をCSVファイルとチャートにしてください。"
        "チャートとCSVファイルを作成したら終了してください。"
    )

    # ワークフローの実行
    events = workflow.workflow.stream(
        {"messages": [("user", message)]},
        {"recursion_limit": 150},  # 最大ステップ数
    )

    # イベントの処理
    for event in events:
        for node_name, node_data in event.items():
            if isinstance(node_data, dict) and "messages" in node_data:
                messages = node_data["messages"]
                if messages:  # メッセージが存在する場合
                    last_message = messages[-1]
                    # メッセージの内容を取得
                    if isinstance(last_message, (tuple, list)):
                        content = last_message[1]  # タプルの場合
                    else:
                        content = last_message.content  # HumanMessageの場合

                    print(f"[{node_name}] {content}")
                    print("----")
