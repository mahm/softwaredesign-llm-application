"""
DSPy ReActを使用したファイル探索エージェント
"""

import os
from pathlib import Path
from typing import Literal
import dspy

from agent_tool_specs import generate_tool_specifications


# ファイル書き込みの基準ディレクトリ（Pythonプロセス起動時のcwd）
_INITIAL_CWD = os.getcwd()


# ============================================================================
# File System Tools
# ============================================================================

# ツールのdocstringはそのままエージェントのプロンプトとして使用されます。
# 変更することにより、エージェントの振る舞い／精度が変わる可能性があります。

def ls_directory(
    path: str = ".",
    recursive: bool = False,
    pattern: str = "*",
) -> str:
    """
    ディレクトリの内容を一覧表示（フィルタリングと再帰オプション付き）

    Args:
        path: 一覧表示するディレクトリパス（デフォルト: カレントディレクトリ）
        recursive: Trueの場合、サブディレクトリを無制限に再帰的に表示
        pattern: 結果をフィルタリングするGlobパターン（デフォルト: "*" 全てのファイル）

    Returns:
        フォーマットされたディレクトリ一覧またはエラーメッセージの文字列

    Examples:
        ls_directory(".")  # カレントディレクトリを一覧表示
        ls_directory(".", recursive=True, pattern="*.py")  # 全てのPythonファイルを再帰的に一覧表示
        ls_directory("src", recursive=True)  # srcディレクトリを再帰的に一覧表示
    """

    try:
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            return f"Error: Path '{path}' does not exist"

        if not path_obj.is_dir():
            return f"Error: Path '{path}' is not a directory"

        results = []

        if recursive:
            # Recursive listing (unlimited depth)
            glob_pattern = f"**/{pattern}"
            items = sorted(path_obj.glob(glob_pattern))

            for item in items:
                # Return absolute path for consistency
                item_type = "DIR" if item.is_dir() else "FILE"
                try:
                    size = item.stat().st_size if item.is_file() else 0
                    results.append(f"{item_type:4s} {size:>10d} {item}")
                except (PermissionError, OSError):
                    results.append(f"{item_type:4s} {'N/A':>10s} {item} (permission denied)")
        else:
            # Non-recursive listing
            items = sorted(path_obj.glob(pattern))

            for item in items:
                # Return absolute path for consistency
                item_type = "DIR" if item.is_dir() else "FILE"
                try:
                    size = item.stat().st_size if item.is_file() else 0
                    results.append(f"{item_type:4s} {size:>10d} {item}")
                except (PermissionError, OSError):
                    results.append(f"{item_type:4s} {'N/A':>10s} {item} (permission denied)")

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
    max_chars: int = 10_000,
    encoding: str = "utf-8",
) -> str:
    """
    ファイルの内容を読み取る（文字数制限とエンコーディング対応）

    Args:
        file_path: 読み取るファイルのパス（絶対パス推奨。ls_directoryの出力をそのまま使用可能）
        max_chars: 読み取る最大文字数（デフォルト: 10000）
        encoding: ファイルエンコーディング（デフォルト: utf-8）

    Returns:
        ファイルの内容またはエラーメッセージの文字列

    Examples:
        read_file("/workspaces/project/README.md")  # 絶対パスで読み取り（推奨）
        read_file("data.txt", max_chars=1000)  # 相対パスも可

    Note:
        ls_directoryは絶対パスを返すので、その出力をそのままfile_pathに渡すことを推奨
    """
    try:
        path_obj = Path(file_path).resolve()

        if not path_obj.exists():
            return f"Error: File '{file_path}' does not exist. Use absolute path (copy the path directly from ls_directory output)"

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


class FileExplorationAgent(dspy.Module):
    """
    ReActベースのファイル探索エージェント

    このエージェントは、ReAct（Reasoning + Acting）フレームワークを使用して、
    ファイルシステムを探索し、ディレクトリ構造を分析し、適切なレポートを生成します。
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

        # Define tool functions
        tools = [ls_directory, read_file, write_file]

        # Generate tool specifications from tool functions
        self.tool_spec = generate_tool_specifications(tools)

        # Create ReAct agent with file system tools
        # DSPy ReAct provides trajectory in Prediction automatically
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
            working_directory: 探索するディレクトリ（相対パスまたは絶対パス）

        Returns:
            レポートと実行トレースを含むPrediction
        """
        # Convert working_directory to absolute path for clarity
        working_directory_abs = str(Path(working_directory).resolve())

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"FILE EXPLORATION AGENT")
            print(f"{'=' * 80}")
            print(f"Task: {task}")
            print(f"Working Directory: {working_directory_abs}")
            print(f"Max Iterations: {self.max_iters}")
            print(f"{'=' * 80}\n")

        # Execute ReAct agent with absolute path
        result = self.agent(
            task=task,
            working_directory=working_directory_abs,
            tool_spec=self.tool_spec
        )

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"FINAL REPORT")
            print(f"{'=' * 80}\n")
            print(result.report)
            print(f"\n{'=' * 80}\n")

        return result
