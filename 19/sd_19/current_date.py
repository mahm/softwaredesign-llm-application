from datetime import datetime


def get_current_date() -> str:
    """現在の日付を取得する"""
    return datetime.now().strftime("%Y年%m月%d日")


current_date = get_current_date()
