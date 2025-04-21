"""
LangGraph Content Creator - アプリケーションエントリポイント

Streamlitを起動して、LangGraphによるコンテンツ生成アプリを実行します。
"""

import os
import sys

import streamlit.web.cli as stcli
from dotenv import load_dotenv


def main():
    """アプリケーションのエントリポイント"""
    # 環境変数の読み込み
    load_dotenv()

    # Streamlitに渡す引数を構築
    app_path = os.path.join(os.path.dirname(__file__), "content_creator/app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.runOnSave=false"]

    # Streamlitの実行
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
