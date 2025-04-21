#!/usr/bin/env python3
"""
LangGraph Content Creator 実行スクリプト
"""

import os
import sys

import streamlit.web.cli as stcli
from dotenv import load_dotenv


def main():
    """
    アプリケーションを実行するエントリポイント
    """
    # 環境変数の読み込み
    load_dotenv()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("エラー: ANTHROPIC_API_KEYが設定されていません。")
        print(".envファイルを作成し、APIキーを設定してください。")
        print("例: ANTHROPIC_API_KEY=your_api_key_here")
        sys.exit(1)

    # Streamlitアプリの実行
    app_path = os.path.join("src", "content_creator", "app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
