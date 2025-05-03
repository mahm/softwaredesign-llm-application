"""
データモデル定義
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """イベントタイプを定義するEnum"""

    OCR_DONE = "ocr_done"
    ACCOUNT_SUGGESTED = "account_suggested"
    SAVE_COMPLETED = "save_completed"
    ERROR = "error"


class CommandType(str, Enum):
    """コマンドタイプを定義するEnum"""

    APPROVE = "approve"
    REGENERATE = "regenerate"


class ReceiptItem(BaseModel):
    """領収書の購入品目を表すモデル"""

    name: str = Field(description="商品名や項目名")
    price: str = Field(description="金額（数字のみ）")


class ReceiptInfoItem(BaseModel):
    """領収書のその他情報を表すモデル"""

    key: str = Field(description="情報の種類（例: 支払方法、伝票番号など）")
    value: str = Field(description="情報の値")


class ReceiptOCRResult(BaseModel):
    """OCR結果の構造化データモデル"""

    raw_text: str = Field(description="領収書から抽出された生テキスト")
    date: str = Field(
        description="領収書の日付（YYYY-MM-DD形式、不明な場合は空文字列）"
    )
    amount: str = Field(
        description="金額（数字のみ、カンマなし、不明な場合は空文字列）"
    )
    shop_name: str = Field(description="店舗・発行元名称（不明な場合は空文字列）")
    items: List[ReceiptItem] = Field(
        description="購入品目のリスト（ある場合のみ）", default_factory=list
    )
    other_info: List[ReceiptInfoItem] = Field(
        description="その他抽出できた情報（領収書番号、支払方法など）",
        default_factory=list,
    )


class AccountInfo(BaseModel):
    """勘定科目情報のデータモデル（CSVにも保存される）"""

    date: str = Field(description="日付（YYYY-MM-DD形式）")
    account: str = Field(description="勘定科目")
    sub_account: str = Field(description="補助科目", default="")
    amount: str = Field(description="金額", default="")
    tax_amount: str = Field(description="消費税額", default="")
    vendor: str = Field(description="取引先", default="")
    invoice_number: str = Field(
        description="インボイス番号（通常Tから始まる文字列）", default=""
    )
    description: str = Field(description="摘要", default="")
    reason: str = Field(description="この勘定科目と判断した理由")
