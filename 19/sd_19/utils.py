from io import StringIO
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from unidiff import PatchSet


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


def apply_patch(original_text: str, diff_text: str) -> str:
    """与えられたテキスト(original_text)にUnified Diff(diff_text)を適用し、結果テキストを返す"""
    lines = original_text.splitlines(
        keepends=True
    )  # 元テキストを行リスト化（改行を維持）
    patch = PatchSet(diff_text)
    for patched_file in patch:
        # Hunkを後ろから順に適用
        for hunk in reversed(list(patched_file)):
            start = hunk.source_start - 1  # 0-indexに変換
            end = start + hunk.source_length
            # Hunk内の出力すべき行を作成（削除行以外を対象。EOFマーカー行も除外）
            new_lines = []
            for line in hunk:
                if line.is_removed:
                    continue  # 削除行は出力しない（元を削除）
                if line.value.startswith("\\ No newline at end of file"):
                    continue  # EOFマーカーはスキップ
                # コンテキスト行および追加行
                new_lines.append(line.value)
            # 元の該当部分を置換
            lines[start:end] = new_lines
    return "".join(lines)


if __name__ == "__main__":
    # 複雑なテストケース：複数のhunk、追加・削除・置換を含む
    original_text = """\
機械学習モデルの開発には以下のステップがあります：
1. データの収集
2. 前処理
3. モデルの選択
4. 学習
5. 評価
6. デプロイ

各ステップで注意点があります。
特に前処理は重要です。
"""

    patch_text = """\
--- draft
+++ draft
@@ -3,1 +3,2 @@
-2. 前処理
+2. データクレンジング
+2.1 前処理
@@ -11,0 +12,1 @@
+データの品質がモデルの性能を大きく左右します。
""".strip()

    # パッチを適用
    result = apply_patch(original_text, patch_text)

    # 期待される結果
    expected = """\
機械学習モデルの開発には以下のステップがあります：
1. データの収集
2. データクレンジング
2.1 前処理
3. モデルの選択
4. 学習
5. 評価
6. デプロイ

各ステップで注意点があります。
特に前処理は重要です。
データの品質がモデルの性能を大きく左右します。
""".strip()

    print("期待される結果:")
    print(expected)
    print("\n実際の結果:")
    print(result)
    print("\n一致:", result == expected)
