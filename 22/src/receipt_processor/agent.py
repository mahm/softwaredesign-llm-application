"""
LangGraph Workflowの定義
"""

import os
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter, interrupt

from src.receipt_processor.account import suggest_account_info
from src.receipt_processor.models import AccountInfo, ReceiptOCRResult
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
            "event": "ocr_done",
            "text": ocr_result.raw_text,
            "structured_data": ocr_result.model_dump(),
        }
    )

    return ocr_result


@task
def generate_account_suggestion(
    ocr_result: ReceiptOCRResult, hint: Optional[str] = None, *, writer: StreamWriter
) -> AccountInfo:
    """
    OCR結果から勘定科目情報を提案するタスク

    Parameters:
    -----------
    ocr_result: ReceiptOCRResult
        OCR処理結果の構造化データ
    hint: Optional[str]
        ユーザーからのフィードバック（あれば）
    writer: StreamWriter
        イベント送信用のStreamWriter

    Returns:
    --------
    AccountInfo
        提案された勘定科目情報
    """
    # 勘定科目提案を取得
    account_info = suggest_account_info(ocr_result, hint)

    # 提案完了イベントを送信
    writer(
        {
            "event": "account_suggested",
            "account_info": account_info.model_dump(),
            "ocr_text": ocr_result.raw_text,
        }
    )

    return account_info


@task
def save_receipt_data(data: Dict[str, Any], *, writer: StreamWriter) -> bool:
    """
    承認されたデータをCSVに保存するタスク

    Parameters:
    -----------
    data: Dict[str, Any]
        保存するデータ
    writer: StreamWriter
        イベント送信用のStreamWriter

    Returns:
    --------
    bool
        保存が成功したかどうか
    """
    # CSVファイルが既に存在する場合はバックアップを作成
    if os.path.exists("tmp/db.csv"):
        backup_csv()

    # デバッグ出力：保存データの確認
    print(f"保存するデータ：{data}")
    for key, value in data.items():
        print(f"  {key}: {value}")

    # データをCSVに保存
    save_success = save_to_csv(data)

    # 保存完了イベントを送信
    writer({"event": "save_complete", "success": save_success, "data": data})

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
    # 各キーの役割:
    # - image_path: 処理中の画像ファイルパス
    # - attempts: フィードバックの試行回数
    # - saved: 保存が完了したかどうか
    # - completed: ワークフロー全体が完了したかどうか
    state = previous or {
        "image_path": image_path,  # 画像パスを保存
        "attempts": 0,  # フィードバック試行回数
        "saved": False,  # 保存完了フラグ
        "completed": False,  # 処理完了フラグ
    }

    # 画像パスが変更された場合は更新
    if image_path != state.get("image_path", ""):
        state["image_path"] = image_path

    # OCRを実行して結果を保存
    ocr_result = process_and_ocr_image(image_path, writer=writer).result()

    # 勘定科目提案を取得
    account_info = generate_account_suggestion(ocr_result, writer=writer).result()

    while True:
        # ユーザーからのフィードバックを待機
        response = interrupt(
            {
                "ocr_result": ocr_result.model_dump(),
                "account_info": account_info.model_dump(),
                "attempts": state.get("attempts", 1),
            }
        )

        # response は必ず dict（コマンドを含む）
        command = response.get("command", "")

        # 承認コマンドの場合
        if command == "approve":
            # 承認された場合はCSVに保存
            final_data = {
                "date": account_info.date,
                "account": account_info.account,
                "sub_account": account_info.sub_account,
                "amount": account_info.amount,
                "tax_amount": account_info.tax_amount,
                "vendor": account_info.vendor,
                "invoice_number": account_info.invoice_number,
                "description": account_info.description,
                "raw_text": ocr_result.raw_text,
            }

            save_success = save_receipt_data(final_data, writer=writer).result()

            # 結果を状態に保存
            state.update(
                {
                    "ocr_result": ocr_result.model_dump(),
                    "account_info": account_info.model_dump(),
                    "attempts": state.get("attempts", 1),
                    "saved": save_success,
                    "completed": True,
                }
            )

            # 完了イベント送信
            writer({"event": "workflow_completed", "success": save_success})
            break

        # 修正コマンドの場合
        elif command == "modify":
            feedback = response.get("feedback", "")
            if feedback:
                # フィードバックを使って勘定科目を再提案
                account_info = generate_account_suggestion(
                    ocr_result, hint=feedback, writer=writer
                ).result()

                # 試行回数を更新
                state["attempts"] = state.get("attempts", 1) + 1

        # 自然言語フィードバックによる再生成
        elif command == "regenerate":
            feedback = response.get("feedback", "")
            if feedback:
                # フィードバックを使って勘定科目を再提案
                account_info = generate_account_suggestion(
                    ocr_result, hint=feedback, writer=writer
                ).result()

                # 試行回数を更新
                state["attempts"] = state.get("attempts", 1) + 1

    return state
