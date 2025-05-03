"""
画像処理とOCR機能
"""

import base64
import mimetypes
import os
import pathlib
import tempfile
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from PIL import Image, ImageEnhance

from src.receipt_processor.models import ReceiptOCRResult


def build_vision_message(image_path: str) -> List[Dict[str, Any]]:
    """
    画像をClaudeのVision APIで使用可能なメッセージ形式に変換

    Parameters:
    -----------
    image_path: str
        画像ファイルのパス

    Returns:
    --------
    List[Dict[str, Any]]
        Claudeに送信するメッセージリスト
    """
    path = pathlib.Path(image_path)
    data = path.read_bytes()
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode()

    # OCRタスク用システムプロンプト
    system_prompt = """\
あなたは領収書OCRシステムです。画像内の領収書からテキストや情報を抽出し、指定された形式で返します。
以下の点に注意してください：
1. 日本語の領収書に特化してください
2. 日付、金額、店舗名は可能な限り抽出してください
3. 日付はYYYY-MM-DD形式に標準化してください
4. 金額は数字のみ（カンマなし）で抽出してください
5. 項目名と金額が対になっている場合は個別の品目として抽出してください
6. その他の重要情報（支払方法や領収書番号など）は種類と値のペアとして抽出してください
7. 生テキストは領収書の全テキストを含めてください
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "この画像から全てのテキストを抽出してください。",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
            ],
        },
    ]


def preprocess_receipt_image(image_path: str) -> str:
    """
    OCR精度向上のための画像前処理

    Parameters:
    -----------
    image_path: str
        元の画像パス

    Returns:
    --------
    str
        処理後の画像の一時ファイルパス
    """
    # 画像を開く
    img = Image.open(image_path)

    # グレースケール変換
    img_gray = img.convert("L")

    # コントラスト強調
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(2.0)  # コントラスト2倍

    # 必要に応じてリサイズ（長辺が1000px以内に）
    max_size = 1000
    if max(img_enhanced.size) > max_size:
        ratio = max_size / max(img_enhanced.size)
        new_size = (
            int(img_enhanced.size[0] * ratio),
            int(img_enhanced.size[1] * ratio),
        )
        img_resized = img_enhanced.resize(
            new_size,
            Image.BICUBIC if hasattr(Image, "BICUBIC") else Image.Resampling.BICUBIC,
        )
    else:
        img_resized = img_enhanced

    # 処理済み画像を一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        img_resized.save(temp_path, format="JPEG", quality=95)

    return temp_path


def ocr_receipt(
    image_path: str, model_name: str = "claude-3-5-haiku-20241022"
) -> ReceiptOCRResult:
    """
    Claude Vision APIを使用して領収書画像からテキストを抽出し、構造化データとして返す

    Parameters:
    -----------
    image_path: str
        処理する画像のファイルパス
    model_name: str
        使用するClaudeモデル名

    Returns:
    --------
    ReceiptOCRResult
        抽出された領収書情報（構造化データ）
    """
    # 前処理を実行
    processed_image_path = preprocess_receipt_image(image_path)

    try:
        # Claude Vision APIを使用するLLMの初期化
        llm = ChatAnthropic(
            model_name=model_name,
            temperature=0,
            timeout=None,
            stop=None,
            max_retries=3,
        )

        # 構造化出力を使って処理
        ocr_chain = llm.with_structured_output(ReceiptOCRResult)

        # OCR処理を実行
        result: ReceiptOCRResult = ocr_chain.invoke(
            build_vision_message(processed_image_path)
        )  # type: ignore

        return result
    finally:
        # 一時ファイルを削除
        if os.path.exists(processed_image_path):
            os.unlink(processed_image_path)
