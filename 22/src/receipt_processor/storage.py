"""
CSV保存機能
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.receipt_processor.models import SavedReceiptData


def save_to_csv(data: Dict[str, Any], csv_path: str = "tmp/db.csv") -> bool:
    """
    データをCSVファイルに保存する

    Parameters:
    -----------
    data: Dict[str, Any]
        保存するデータ（日付、金額、勘定科目情報など）
    csv_path: str
        CSVファイルのパス

    Returns:
    --------
    bool
        保存が成功したかどうか
    """
    # 保存先ディレクトリが存在しない場合は作成
    csv_dir = os.path.dirname(csv_path)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # 検証用にpydanticモデルを使用
    receipt_data = SavedReceiptData(**data)

    # ファイルが存在するかチェック
    file_exists = os.path.isfile(csv_path)

    # CSVに保存（新規作成または追記）
    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
            fieldnames = [
                "date",
                "account",
                "sub_account",
                "amount",
                "tax_amount",
                "vendor",
                "invoice_number",
                "description",
                "raw_text",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # ファイルが存在しない場合はヘッダーを書き込む
            if not file_exists:
                writer.writeheader()

            # データを書き込む
            writer.writerow(receipt_data.model_dump())

        return True
    except Exception as e:
        print(f"CSV保存エラー: {e}")
        return False


def get_saved_receipts(csv_path: str = "tmp/db.csv") -> List[Dict[str, Any]]:
    """
    保存された領収書データを取得する

    Parameters:
    -----------
    csv_path: str
        CSVファイルのパス

    Returns:
    --------
    List[Dict[str, Any]]
        保存されたデータのリスト
    """
    if not os.path.exists(csv_path):
        return []

    try:
        # pandasでCSVを読み込む
        df = pd.read_csv(csv_path, encoding="utf-8")
        # DataFrame -> Dict変換
        receipts = df.to_dict(orient="records")
        return receipts
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        return []


def backup_csv(csv_path: str = "tmp/db.csv") -> bool:
    """
    CSVファイルのバックアップを作成する

    Parameters:
    -----------
    csv_path: str
        バックアップするCSVファイルのパス

    Returns:
    --------
    bool
        バックアップが成功したかどうか
    """
    if not os.path.exists(csv_path):
        return False

    try:
        # 元のファイルパスと拡張子を取得
        path = Path(csv_path)
        backup_path = path.with_name(f"{path.stem}_backup{path.suffix}")

        # ファイルをコピー
        import shutil

        shutil.copy2(csv_path, backup_path)
        return True
    except Exception as e:
        print(f"バックアップエラー: {e}")
        return False
