#!/usr/bin/env python
"""
領収書OCRエージェント実行スクリプト
"""
import os
import sys
from pathlib import Path

import streamlit.web.cli as stcli
from dotenv import load_dotenv


def main() -> None:
    """アプリケーションのメイン関数"""
    # 環境変数の読み込み
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)

    # 環境変数チェック
    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"エラー: 以下の環境変数が設定されていません: {', '.join(missing_vars)}")
        print("実行前に.envファイルを作成するか、環境変数を設定してください。")
        sys.exit(1)

    # プロジェクトルートディレクトリをPYTHONPATHに追加
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))

    # tmp ディレクトリがない場合は作成
    tmp_dir = project_root / "tmp"
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)

    # アプリケーションファイルへのパス（直接ファイルパスを指定）
    app_path = str(project_root / "src" / "receipt_processor" / "app.py")

    # Streamlitを直接実行（ファイルパスを指定して実行）
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=localhost",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
