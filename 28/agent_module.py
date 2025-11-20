"""
DSPy ReActを使用したファイル探索エージェントモジュール

このモジュールは、ディレクトリをナビゲートし、ファイルを読み取り、
ディレクトリ構造に関する包括的なレポートを生成するファイルシステム探索エージェントを実装します。
"""

import os
import functools
from pathlib import Path
from typing import Literal, Optional
import dspy

from agent_tool_specs import generate_tool_specifications


# ファイル書き込みの基準ディレクトリ（Pythonプロセス起動時のcwd）
_INITIAL_CWD = os.getcwd()


# ============================================================================
# File System Tools
# ============================================================================


def ls_directory(
    path: str = ".",
    recursive: bool = False,
    pattern: str = "*",
    max_depth: Optional[int] = None,
) -> str:
    """
    ディレクトリの内容を一覧表示（フィルタリングと再帰オプション付き）

    Args:
        path: 一覧表示するディレクトリパス（デフォルト: カレントディレクトリ）
        recursive: Trueの場合、サブディレクトリを再帰的に表示
        pattern: 結果をフィルタリングするGlobパターン（デフォルト: "*" 全てのファイル）
        max_depth: 再帰表示の最大深度（None = 無制限）

    Returns:
        フォーマットされたディレクトリ一覧またはエラーメッセージの文字列

    Examples:
        ls_directory(".")  # カレントディレクトリを一覧表示
        ls_directory(".", recursive=True, pattern="*.py")  # 全てのPythonファイルを一覧表示
        ls_directory("src", max_depth=2)  # 最大2階層まで表示
    """
    try:
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            return f"Error: Path '{path}' does not exist"

        if not path_obj.is_dir():
            return f"Error: Path '{path}' is not a directory"

        results = []

        if recursive:
            # Recursive listing
            if max_depth is None:
                glob_pattern = f"**/{pattern}"
            else:
                # Build pattern for max depth
                depth_patterns = [pattern]
                for d in range(1, max_depth + 1):
                    depth_patterns.append("/".join(["*"] * d) + f"/{pattern}")

                # Use first depth pattern for simplicity in this implementation
                glob_pattern = f"**/{pattern}"

            items = sorted(path_obj.glob(glob_pattern))

            # Filter by depth if specified
            if max_depth is not None:
                base_depth = len(path_obj.parts)
                items = [
                    item for item in items
                    if len(item.parts) - base_depth <= max_depth
                ]

            for item in items:
                rel_path = item.relative_to(path_obj)
                item_type = "DIR" if item.is_dir() else "FILE"
                try:
                    size = item.stat().st_size if item.is_file() else 0
                    results.append(f"{item_type:4s} {size:>10d} {rel_path}")
                except (PermissionError, OSError):
                    results.append(f"{item_type:4s} {'N/A':>10s} {rel_path} (permission denied)")
        else:
            # Non-recursive listing
            items = sorted(path_obj.glob(pattern))

            for item in items:
                item_type = "DIR" if item.is_dir() else "FILE"
                try:
                    size = item.stat().st_size if item.is_file() else 0
                    results.append(f"{item_type:4s} {size:>10d} {item.name}")
                except (PermissionError, OSError):
                    results.append(f"{item_type:4s} {'N/A':>10s} {item.name} (permission denied)")

        if not results:
            return f"No items found matching pattern '{pattern}' in '{path}'"

        header = f"Listing: {path} (pattern: {pattern}, recursive: {recursive})\n"
        header += f"{'Type':<4s} {'Size':>10s} {'Name'}\n"
        header += "-" * 60

        return header + "\n" + "\n".join(results)

    except PermissionError:
        return f"Error: Permission denied accessing '{path}'"
    except OSError as e:
        return f"Error: OS error accessing '{path}': {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error: {str(e)}"


def read_file(
    file_path: str,
    max_chars: int = 5000,
    encoding: str = "utf-8",
) -> str:
    """
    ファイルの内容を読み取る（文字数制限とエンコーディング対応）

    Args:
        file_path: 読み取るファイルのパス
        max_chars: 読み取る最大文字数（デフォルト: 5000）
        encoding: ファイルエンコーディング（デフォルト: utf-8）

    Returns:
        ファイルの内容またはエラーメッセージの文字列

    Examples:
        read_file("README.md")  # 最初の5000文字を読み取り
        read_file("data.txt", max_chars=1000)  # 最初の1000文字を読み取り
    """
    try:
        path_obj = Path(file_path).resolve()

        if not path_obj.exists():
            return f"Error: File '{file_path}' does not exist"

        if not path_obj.is_file():
            return f"Error: Path '{file_path}' is not a file"

        # Check file size
        file_size = path_obj.stat().st_size

        try:
            with open(path_obj, 'r', encoding=encoding) as f:
                content = f.read(max_chars)

            if len(content) >= max_chars:
                content += f"\n... (truncated at {max_chars} characters, file size: {file_size} bytes)"

            return f"File: {file_path} ({file_size} bytes)\n{'=' * 60}\n{content}"

        except UnicodeDecodeError:
            return f"Error: Cannot decode '{file_path}' with encoding '{encoding}'. File may be binary."

    except PermissionError:
        return f"Error: Permission denied reading '{file_path}'"
    except OSError as e:
        return f"Error: OS error reading '{file_path}': {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error: {str(e)}"


def write_file(
    file_path: str,
    content: str,
    mode: Literal["overwrite", "append", "create_new"] = "overwrite",
) -> str:
    """
    ファイルにコンテンツを書き込む（複数のモードをサポート）

    重要: 相対パスはPythonスクリプト起動ディレクトリに対して解決されます。
    探索中のディレクトリではなく、コマンドを実行したディレクトリに作成されます。

    Args:
        file_path: 書き込むファイルのパス（相対パスまたは絶対パス）
        content: 書き込むコンテンツ
        mode: 書き込みモード - "overwrite"（上書き）, "append"（追記）, "create_new"（新規作成のみ）

    Returns:
        成功またはエラーメッセージ

    Examples:
        write_file("report.txt", "Analysis results")  # スクリプト起動ディレクトリにreport.txtを作成
        write_file("logs/output.txt", "New entry", mode="append")  # logs/output.txtに追記
    """
    try:
        # 相対パスの場合、スクリプト起動ディレクトリ（_INITIAL_CWD）に対して解決
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = Path(_INITIAL_CWD) / path_obj

        path_obj = path_obj.resolve()

        # モード制約のチェック
        if mode == "create_new" and path_obj.exists():
            return f"Error: File '{file_path}' already exists (mode=create_new)"

        # ファイルオープンモードの決定
        if mode == "append":
            file_mode = 'a'
        else:  # overwrite or create_new
            file_mode = 'w'

        # 必要に応じて親ディレクトリを作成
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, file_mode, encoding='utf-8') as f:
            f.write(content)

        action = "appended to" if mode == "append" else "written to"
        bytes_written = len(content.encode('utf-8'))

        return f"Success: {bytes_written} bytes {action} '{path_obj}'"

    except PermissionError:
        return f"Error: Permission denied writing to '{file_path}'"
    except OSError as e:
        if "No space left" in str(e) or "Disk quota exceeded" in str(e):
            return f"Error: No disk space available: {str(e)}"
        return f"Error: OS error writing to '{file_path}': {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error: {str(e)}"


# ============================================================================
# DSPy Agent Implementation
# ============================================================================


class FileExplorationSignature(dspy.Signature):
    """
    ファイル探索タスクのシグネチャ

    エージェントは、利用可能なファイルシステムツール（ls_directory, read_file, write_file）を使用して、
    指定されたディレクトリを分析し、要求されたタスクを完了する必要があります。
    """

    task: str = dspy.InputField(
        desc="タスクの説明（例: 'ディレクトリ構造を分析してレポートを作成'）"
    )
    working_directory: str = dspy.InputField(
        desc="探索する作業ディレクトリのパス"
    )
    tool_spec: str = dspy.InputField(
        desc="利用可能なツールの仕様（引数の名前、型、デフォルト値を含む詳細な説明）"
    )
    report: str = dspy.OutputField(
        desc="発見事項、分析、洞察を含む包括的なレポート"
    )


class VerboseToolWrapper:
    """ツール呼び出しに詳細ログを追加するラッパー"""

    def __init__(self, tool, verbose=True):
        self.tool = tool
        self.verbose = verbose
        # functools.update_wrapperで全メタデータを自動コピー
        functools.update_wrapper(self, tool)

    def __call__(self, *args, **kwargs):
        """
        ツールを普通のPython関数として呼び出す

        DSPy ReActはツールを通常のPython関数として呼び出します。
        例: tool(city="Tokyo") または tool("/path/to/dir", recursive=True)

        DSPyが試行的に args={}, kwargs={...} 形式で呼び出す場合にも対応します。
        """
        # DSPyが args/kwargs という名前のキーワード引数を渡す場合に対応
        # これはDSPyの探索フェーズで発生する可能性がある
        if 'args' in kwargs and 'kwargs' in kwargs and len(args) == 0:
            # DSPyの探索形式: tool(args=[], kwargs={...})
            actual_args = kwargs.pop('args', [])
            actual_kwargs = kwargs.pop('kwargs', {})
            # argsとkwargsを展開
            if isinstance(actual_args, (list, tuple)):
                args = tuple(actual_args)
            if isinstance(actual_kwargs, dict):
                kwargs = actual_kwargs

        if self.verbose:
            # 引数の表示（長い文字列は切り詰め）
            args_repr = []
            for a in args:
                if isinstance(a, str) and len(a) > 50:
                    args_repr.append(f"'{a[:50]}...'")
                else:
                    args_repr.append(repr(a))

            kwargs_repr = []
            for k, v in kwargs.items():
                if isinstance(v, str) and len(v) > 50:
                    kwargs_repr.append(f"{k}='{v[:50]}...'")
                else:
                    kwargs_repr.append(f"{k}={repr(v)}")

            all_args = ", ".join(args_repr + kwargs_repr)
            print(f"\n[ACTION] {self.__name__}({all_args})")

        try:
            # ツールを直接呼び出す
            result = self.tool(*args, **kwargs)

            if self.verbose:
                result_str = str(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + f"\n... (truncated, total: {len(result_str)} chars)"
                print(f"[OBSERVATION] {result_str}\n")

            return result

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] {type(e).__name__}: {str(e)}\n")
            raise


class FileExplorationAgent(dspy.Module):
    """
    ReActベースのファイル探索エージェント

    このエージェントは、ReAct（Reasoning + Acting）フレームワークを使用して、
    ファイルシステムを探索し、ディレクトリ構造を分析し、包括的なレポートを生成します。
    """

    def __init__(self, max_iters: int = 10, verbose: bool = True):
        """
        ファイル探索エージェントを初期化

        Args:
            max_iters: ReActの最大反復回数
            verbose: Trueの場合、詳細な実行トレースを出力
        """
        super().__init__()
        self.max_iters = max_iters
        self.verbose = verbose

        # Define raw tool functions (before verbose wrapping)
        raw_tools = [ls_directory, read_file, write_file]

        # Generate tool specifications from raw tool functions
        self.tool_spec = generate_tool_specifications(raw_tools)

        # Wrap tools with verbose logging if enabled
        tools = raw_tools
        if verbose:
            tools = [VerboseToolWrapper(tool, verbose=True) for tool in raw_tools]

        # Create ReAct agent with file system tools
        self.agent = dspy.ReAct(
            signature=FileExplorationSignature,
            tools=tools,
            max_iters=max_iters,
        )

    def forward(self, task: str, working_directory: str = ".") -> dspy.Prediction:
        """
        ファイル探索タスクを実行

        Args:
            task: タスクの説明
            working_directory: 探索するディレクトリ

        Returns:
            レポートと実行トレースを含むPrediction
        """
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"FILE EXPLORATION AGENT")
            print(f"{'=' * 80}")
            print(f"Task: {task}")
            print(f"Working Directory: {working_directory}")
            print(f"Max Iterations: {self.max_iters}")
            print(f"{'=' * 80}\n")

        # Execute ReAct agent (tool calls will be logged by VerboseToolWrapper)
        result = self.agent(
            task=task,
            working_directory=working_directory,
            tool_spec=self.tool_spec
        )

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"FINAL REPORT")
            print(f"{'=' * 80}\n")
            print(result.report)
            print(f"\n{'=' * 80}\n")

        return result
