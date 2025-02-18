import difflib
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate


def load_prompt(chain_name: str) -> ChatPromptTemplate:
    """
    指定されたチェーンのプロンプトを読み込む

    Args:
        chain_name: チェーンの名前（例: "writer", "refiner", "evaluator"）
    Returns:
        プロンプトの内容
    """
    prompt_dir = Path(__file__).parent / "prompts" / chain_name
    if not prompt_dir.exists():
        raise FileNotFoundError(f"プロンプトディレクトリが見つかりません: {prompt_dir}")

    with open(prompt_dir / "system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open(prompt_dir / "human.txt", "r", encoding="utf-8") as f:
        human_prompt = f.read().strip()

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
    return ChatPromptTemplate.from_messages(messages)


def parse_unified_diff(diff_text: str) -> List[tuple[str, str]]:
    """
    統一diff形式のテキストを解析する

    Args:
        diff_text: 統一diff形式のテキスト
    Returns:
        変更のリスト。各要素は(操作, 行)のタプル。
        操作は '+' (追加), '-' (削除), ' ' (変更なし) のいずれか。
    """
    changes = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue  # ヘッダー行をスキップ
        if not line:
            continue  # 空行をスキップ
        if line.startswith("+"):
            changes.append(("+", line[1:]))
        elif line.startswith("-"):
            changes.append(("-", line[1:]))
        else:
            changes.append((" ", line))
    return changes


def apply_diff(original: str, diff: str) -> str:
    """
    Unified Diff形式の差分を元の文章に適用する

    Args:
        original: 元の文章
        diff: Unified Diff形式の差分
    Returns:
        差分を適用した文章
    """
    # diffの前後にある可能性のあるマークダウンのコードブロックを除去
    diff = diff.replace("```diff\n", "").replace("\n```", "")

    # 元の文章を行に分割
    lines = original.split("\n")
    result = lines.copy()

    # diffの各行を処理
    current_line = 0
    for line in diff.split("\n"):
        # ヘッダー行をスキップ
        if line.startswith("---") or line.startswith("+++"):
            continue

        # ハンク（変更箇所）のヘッダーを解析
        if line.startswith("@@"):
            parts = line.split(" ")
            # 例: @@ -1,3 +1,3 @@ から1を取得
            current_line = int(parts[1].split(",")[0][1:]) - 1
            continue

        # 変更行を処理
        if line.startswith("+"):
            # 追加行
            result.insert(current_line, line[1:])
            current_line += 1
        elif line.startswith("-"):
            # 削除行
            if current_line < len(result):
                result.pop(current_line)
        else:
            # 変更なしの行
            current_line += 1

    return "\n".join(result)
