from langchain_core.output_parsers import StrOutputParser
from retry import retry

from ..llm import get_llm
from ..utils import apply_patch, load_prompt


@retry(exceptions=Exception, tries=3, delay=1, backoff=2)
def _generate_patch(draft: str, style: str, search_results: str) -> str:
    chain = load_prompt("refiner") | get_llm(model="gpt-4o") | StrOutputParser()
    diff_str = chain.invoke(
        {"draft": draft, "style": style, "search_results": search_results}
    )

    # パッチの検証
    apply_patch(draft, diff_str)  # 無効なパッチの場合は例外が発生

    return diff_str


def run(draft: str, style: str, search_results: str) -> str:
    """
    記事を指定されたスタイルで改善する

    Args:
        draft: 改善対象の記事内容
        style: 改善のスタイル（"保守的"、"中間"、"積極的"）
        search_results: spawn_candidatesで実行された検索結果
    Returns:
        str: 変更内容（変更が必要な行のみ）。生成に失敗した場合は空文字列
    """
    try:
        return _generate_patch(draft, style, search_results)
    except Exception:
        return "※ パッチの生成に失敗しました。"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    draft = """\
AI技術は急速に発展しています。
機械学習の応用範囲は広がっています。
深層学習は画像認識で優れた成果を上げています。
""".strip()

    # 2つの異なる検索結果でテスト
    search_results = """\
・2023年のAI市場規模は前年比30%増
・機械学習は医療診断や自動運転などで実用化が進む
・深層学習の精度は人間を上回るケースも
""".strip()

    print("=== 改善前 ===")
    print(draft)

    print("\n=== 生成されたパッチ ===")
    patch_str = run(draft, "中間", search_results)
    print(patch_str)

    print("\n=== 適用後 ===")
    print(apply_patch(draft, patch_str))
