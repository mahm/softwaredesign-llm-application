"""
LangGraph Workflowの定義
"""

import os
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter, interrupt

from src.receipt_processor.account import suggest_account_info
from src.receipt_processor.constants import CSV_FILE_PATH
from src.receipt_processor.models import (
    AccountInfo,
    CommandType,
    EventType,
    Feedback,
    ReceiptOCRResult,
)
from src.receipt_processor.storage import backup_csv, save_to_csv
from src.receipt_processor.vision import ocr_receipt


@task
def process_and_ocr_image(image_path: str, *, writer: StreamWriter) -> ReceiptOCRResult:
    """
    画像の前処理とOCR処理を行う統合タスク

    Parameters:
    -----------
    image_path: str
        処理する画像のファイルパス
    writer: StreamWriter
        イベント送信用のStreamWriter

    Returns:
    --------
    ReceiptOCRResult
        抽出された領収書情報（構造化データ）
    """
    # 画像パスが存在するかチェック
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # OCR処理を実行
    ocr_result = ocr_receipt(image_path)

    # OCR完了イベントを送信
    writer(
        {
            "event": EventType.OCR_DONE,
            "text": ocr_result.raw_text,
            "structured_data": ocr_result.model_dump(),
        }
    )

    return ocr_result


@task
def generate_account_suggestion(
    ocr_result: ReceiptOCRResult,
    feedback_history: Optional[List[str]] = None,
    *,
    writer: StreamWriter,
) -> AccountInfo:
    """
    OCR結果から勘定科目情報を提案するタスク

    Parameters:
    -----------
    ocr_result: ReceiptOCRResult
        OCR処理結果の構造化データ
    feedback_history: Optional[List[str]]
        これまでのユーザーフィードバック履歴
    writer: StreamWriter
        イベント送信用のStreamWriter

    Returns:
    --------
    AccountInfo
        提案された勘定科目情報
    """
    # フィードバック履歴がある場合はフィードバックとして使用
    combined_feedback = None
    if feedback_history and len(feedback_history) > 0:
        # すべてのフィードバックを結合して渡す
        combined_feedback = " ".join(
            [f"フィードバック{i+1}: {fb}" for i, fb in enumerate(feedback_history)]
        )

    # 勘定科目提案を取得
    account_info = suggest_account_info(ocr_result, combined_feedback)

    # 提案完了イベントを送信
    writer(
        {
            "event": EventType.ACCOUNT_SUGGESTED,
            "account_info": account_info.model_dump(),
            "ocr_text": ocr_result.raw_text,
        }
    )

    return account_info


@task
def save_receipt_data(data: AccountInfo, *, writer: StreamWriter) -> bool:
    """
    承認されたデータをCSVに保存するタスク

    Parameters:
    -----------
    data: AccountInfo
        保存するデータ
    writer: StreamWriter
        イベント送信用のStreamWriter

    Returns:
    --------
    bool
        保存が成功したかどうか
    """
    # CSVファイルが既に存在する場合はバックアップを作成
    if os.path.exists(CSV_FILE_PATH):
        backup_csv()

    # データをCSVに保存
    save_success = save_to_csv(data)

    # 保存完了イベントを送信
    writer(
        {
            "event": EventType.SAVE_COMPLETED,
            "account_info": data.model_dump(),
        }
    )

    return save_success


@entrypoint(checkpointer=MemorySaver())
def receipt_workflow(
    image_path: str,
    *,
    previous: Any = None,
    writer: StreamWriter,
) -> Dict[str, Any]:
    """
    領収書OCRと勘定科目提案のワークフロー

    Parameters:
    -----------
    image_path: str
        処理する画像のファイルパス
    previous: Any
        前回の状態データ
    writer: StreamWriter
        イベント送信用のStreamWriter

    Returns:
    --------
    Dict[str, Any]
        更新された状態
    """
    # 状態の初期化または復元
    state = previous or {
        "image_path": image_path,  # 処理中の画像ファイルパス
        "feedback_history": [],  # ユーザーからのフィードバック履歴
        "completed": False,  # ワークフロー完了フラグ
    }

    # 画像パスが変更された場合は更新
    if image_path != state.get("image_path", ""):
        state["image_path"] = image_path
        # 新しい画像の場合はフィードバック履歴をリセット
        state["feedback_history"] = []

    # OCRを実行して結果を保存
    ocr_result = process_and_ocr_image(image_path, writer=writer).result()

    # 勘定科目提案を取得（フィードバック履歴があれば利用）
    account_info = generate_account_suggestion(
        ocr_result, feedback_history=state.get("feedback_history", []), writer=writer
    ).result()

    while True:
        # ユーザーからのフィードバックを待機
        response = interrupt(
            {
                "ocr_result": ocr_result.model_dump(),
                "account_info": account_info.model_dump(),
                "feedback_count": len(state.get("feedback_history", [])),
            }
        )

        feedback: Feedback = Feedback.model_validate(response)

        # 承認コマンドの場合
        if feedback.command == CommandType.APPROVE:
            # CSVに保存
            save_receipt_data(account_info, writer=writer).result()

            # 状態を更新して完了とマーク
            state.update(
                {
                    "ocr_result": ocr_result.model_dump(),
                    "account_info": account_info.model_dump(),
                    "completed": True,
                }
            )

            break

        # フィードバックを受け取った場合
        elif feedback.command == CommandType.REGENERATE:
            feedback = feedback.feedback

            if feedback:
                # フィードバック履歴に追加
                state["feedback_history"] = state.get("feedback_history", []) + [
                    feedback
                ]
                # フィードバックを使って勘定科目を再提案
                account_info = generate_account_suggestion(
                    ocr_result,
                    feedback_history=state["feedback_history"],
                    writer=writer,
                ).result()

        # 未知のコマンドが渡された場合
        else:
            # エラーイベントを送信
            writer(
                {
                    "event": EventType.ERROR,
                    "message": f"不正なコマンドです: {feedback.command}。'approve'または'regenerate'を使用してください。",
                    "command": feedback.command,
                }
            )

    return state
