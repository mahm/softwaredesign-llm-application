from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

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


class FileReader:
    """ファイル読み込みを行うクラス

    生成されたCSVファイルやチャート画像を読み込み、
    その内容を検証するために使用します。
    """

    def __init__(self, timestamp: str) -> None:
        """FileReaderを初期化します。

        Args:
            timestamp: 出力ファイルの保存に使用するタイムスタンプ
        """
        self.timestamp = timestamp

    def create_tool(self) -> BaseTool:
        """ファイル読み込みツールを生成します。

        Returns:
            BaseTool: ファイル読み込み用のツール
        """
        timestamp = self.timestamp

        @tool
        def file_reader(file_path: str) -> str:
            """ファイルの内容を読み込むツール

            Args:
                file_path: 読み込むファイルのパス
                         - 完全なパス（output/{timestamp}/data/file.csv）
                         - 相対パス（data/file.csv）のどちらも指定可能

            Returns:
                str: ファイルの内容
                     CSVの場合: 先頭10行のデータ
                     画像の場合: ファイルの存在確認と基本情報
            """
            import os
            from pathlib import Path

            def read_file(path: Path) -> Optional[str]:
                """ファイルを読み込み、内容を返します。

                Args:
                    path: 読み込むファイルのパス

                Returns:
                    Optional[str]: ファイルの内容。ファイルが存在しない場合はNone
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
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name=self.config.name
        )
        return Command(update={"messages": result["messages"]}, goto=next_node)


def _read_prompt(file_name: str, **kwargs) -> str:
    """プロンプトファイルを読み込み、必要に応じて変数を置換します。

    Args:
        file_name: プロンプトファイルの名前（拡張子含む）
        **kwargs: プロンプト内の変数を置換するための辞書

    Returns:
        str: 読み込んだプロンプト
    """
    with open(f"prompts/{file_name}", "r", encoding="utf-8") as f:
        prompt = f.read()

    # 変数の置換
    if kwargs:
        prompt = prompt.format(**kwargs)

    return prompt


class ResearchAgent(Agent):
    """リサーチを行うエージェント

    Tavilyの検索APIを使用して情報を収集し、
    データをCSV形式でまとめるようにCodeGeneratorに依頼します。
    """

    def __init__(self) -> None:
        """リサーチエージェントを初期化します。"""
        prompt = _read_prompt("researcher.prompt")
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


class ReflectionAgent(Agent):
    """データの検証と品質チェックを行うエージェント

    リサーチ結果とデータ生成結果を検証し、以下を確認します：
    1. 情報の正確性と信頼性
    2. データの完全性
    3. 生成されたデータと元の情報との整合性
    """

    def __init__(self, file_reader: FileReader) -> None:
        """リフレクションエージェントを初期化します。

        Args:
            file_reader: タイムスタンプ付きのファイル読み込みツール
        """
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
        """エージェントを実行し、検証結果に基づいて次のノードを決定します。

        Args:
            state: 現在のメッセージ状態
            next_node: デフォルトの次のノード（この場合は使用しません）

        Returns:
            Command: 次のノードとメッセージの更新情報を含むコマンド
                - 検証OKの場合: code_generatorに進む
                - 検証NGの場合: researcherに差し戻し
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
            goto = "researcher"
            content += (
                "\n\n次のノードが不明確です。正しい形式で次のノードを指定してください。"
            )

        result["messages"][-1] = HumanMessage(content=content, name=self.config.name)
        return Command(update={"messages": result["messages"]}, goto=goto)


class WorkflowGraph:
    """ワークフローを管理するクラス

    ResearchAgent、CodeGenerator、ReflectionAgentを組み合わせて、
    データの収集から可視化、検証までの一連の処理を管理します。

    Attributes:
        timestamp (str): 実行時のタイムスタンプ
        research_agent (ResearchAgent): リサーチを行うエージェント
        code_generator (CodeGenerator): コード生成を行うエージェント
        reflection_agent (ReflectionAgent): データ検証を行うエージェント
        workflow (Any): コンパイル済みのワークフローグラフ
    """

    def __init__(self) -> None:
        """ワークフローを初期化します。"""
        self.timestamp = self._get_timestamp()
        self._ensure_output_dirs()
        self.research_agent = ResearchAgent()
        self.code_generator = CodeGenerator(self.timestamp)
        self.file_reader = FileReader(self.timestamp)
        self.reflection_agent = ReflectionAgent(self.file_reader)
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

        以下の順序で実行されるワークフローを構築します：
        1. ResearchAgent: データの収集
        2. ReflectionAgent: 検証
        3. CodeGenerator: データの処理と可視化
        4. ReflectionAgent: 最終検証

        各ノードはCommandオブジェクトを返し、動的に次のノードを決定します。

        Returns:
            StateGraph: コンパイル済みのワークフローグラフ
        """
        workflow = StateGraph(MessagesState)

        # ノードの追加
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("code_generator", self._code_node)

        # 開始ノードの設定
        workflow.add_edge(START, "researcher")

        return workflow.compile()

    def _research_node(self, state: MessagesState) -> Command[Literal["reflection"]]:
        """リサーチノードを実行します。

        Args:
            state: 現在のメッセージ状態

        Returns:
            Command: 次のノード（reflection）とメッセージの更新情報を含むコマンド
        """
        return self.research_agent.run(state, "reflection")

    def _reflection_node(
        self, state: MessagesState
    ) -> Command[Literal["researcher", "code_generator", END]]:
        """リフレクションノードを実行します。

        リサーチ結果を検証し、次のノードを決定します：
        - 問題がある場合: フィードバックと共にresearcherに差し戻し
        - 問題がない場合: code_generatorに進む
        - 最終確認OKの場合: ENDに進む

        Args:
            state: 現在のメッセージ状態

        Returns:
            Command: 次のノード（researcher、code_generator、またはEND）とメッセージの更新情報を含むコマンド
        """
        return self.reflection_agent.run(state, "code_generator")

    def _code_node(self, state: MessagesState) -> Command[Literal["reflection"]]:
        """コード生成ノードを実行します。

        データの処理と可視化を行い、検証のためにReflectionAgentに進みます。

        Args:
            state: 現在のメッセージ状態

        Returns:
            Command: 次のノード（reflection）とメッセージの更新情報を含むコマンド
        """
        return self.code_generator.run(state, "reflection")

    def run(self, message: str) -> None:
        """ワークフローを実行します。

        Args:
            message: ユーザーからの初期メッセージ。
                    データの収集と可視化に関する要求を含みます。

        Note:
            イベントの構造:
            {
                "実行ノード名": {
                    "messages": [メッセージのリスト]
                }
            }
        """
        events = self.workflow.stream(
            {"messages": [("user", message)]},
            {"recursion_limit": 150},  # 最大ステップ数
        )
        for event in events:
            # イベントからメッセージを取得
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    workflow = WorkflowGraph()
    workflow.run(
        "2024年の1年間の日経平均推移をCSVファイルとチャートにしてください。"
        "チャートとCSVファイルを作成したら終了してください。"
    )
