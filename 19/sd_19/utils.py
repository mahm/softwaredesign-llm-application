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


def apply_patch(original_text: str, patch_text: str) -> str:
    """
    テキストにパッチを適用する（-u0形式のパッチに対応）
    Args:
        original_text: 元のテキスト
        patch_text: パッチの内容（-u0形式）
    Returns:
        str: パッチ適用後のテキスト
    """
    # 元テキストを行リストに変換
    original_lines = original_text.splitlines(keepends=True)

    # パッチテキストからPatchSetオブジェクトを作成
    patch = PatchSet(StringIO(patch_text))

    # パッチファイル内の最初のファイルパッチを処理
    if not patch:
        return original_text

    patched_file = patch[0]
    patched_lines = original_lines[:]
    offset = 0

    # 各hunk（変更ブロック）に対して処理
    for hunk in patched_file:
        # hunk.source_startは1-indexedなので0-indexedに変換し、これまでのoffsetを加味
        start = hunk.source_start - 1 + offset
        # 元テキスト側で削除対象となる行数
        old_block_len = hunk.source_length

        # hunk の新しいブロック（target block）を構築：共通行と追加行の両方を採用
        new_block = []
        for line in hunk:
            if line.is_context or line.is_added:
                new_block.append(line.value)

        # 対象ブロックを new_block で置換
        patched_lines[start : start + old_block_len] = new_block
        # offsetを更新：新ブロックの行数と元ブロックの行数の差分
        offset += len(new_block) - old_block_len

    # 結果を文字列として返す
    return "".join(patched_lines)


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
