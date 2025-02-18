import os

from dotenv import load_dotenv

from .workflows.web_explorer import create_web_explorer_workflow


def main():
    """メイン関数"""
    # 環境変数の読み込み
    load_dotenv()

    # APIキーの確認
    required_keys = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"Error: 以下のAPIキーが設定されていません: {', '.join(missing_keys)}")
        return

    # ワークフローの作成
    workflow = create_web_explorer_workflow()

    # 初期状態の設定
    initial_state = {
        "query": "AI技術の最新動向について、特に生成AIの発展と応用に焦点を当てて説明してください。",
    }

    try:
        # ワークフローの実行
        final_state = workflow.compile().invoke(initial_state)

        # 結果の出力
        if "final_article" in final_state:
            print("\n=== 生成された記事 ===\n")
            print(final_state["final_article"])
        else:
            print("Error: 記事の生成に失敗しました。")

    except Exception as e:
        print(f"Error: ワークフローの実行中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
