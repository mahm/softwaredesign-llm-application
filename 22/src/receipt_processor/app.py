"""
Streamlitアプリケーション
"""

import os
import uuid
from typing import Any, Dict, Generator, Optional, Union

import streamlit as st
from langgraph.types import Command

from src.receipt_processor.agent import receipt_workflow
from src.receipt_processor.models import CommandType, EventType
from src.receipt_processor.storage import backup_csv, get_saved_receipts
from src.receipt_processor.ui_components import (
    account_info_editor,
    display_action_buttons,
    display_loading_spinner,
    display_ocr_text,
    display_receipt_history,
    display_success_message,
    handle_image_input,
    setup_page,
)


def init_session_state() -> None:
    """セッション状態の初期化"""
    # スレッドIDの初期化
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # ワークフロー状態の初期化
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = "idle"  # idle, processing, feedback, complete

    # OCRデータの初期化
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "ocr_result" not in st.session_state:
        st.session_state.ocr_result = {}

    # 勘定科目情報の初期化
    if "account_info" not in st.session_state:
        st.session_state.account_info = {}

    # イベントと割り込みの初期化
    if "events" not in st.session_state:
        st.session_state.events = []
    if "current_interrupt" not in st.session_state:
        st.session_state.current_interrupt = None

    # フィールド変更検出用
    if "field_values_changed" not in st.session_state:
        st.session_state.field_values_changed = False

    # 一時ファイル管理
    if "temp_files" not in st.session_state:
        st.session_state.temp_files = []

    # 表示モード
    if "display_mode" not in st.session_state:
        st.session_state.display_mode = "input"  # input, history


def stream_handler(data: Dict[str, Any]) -> None:
    """
    StreamWriter経由で送信されるイベントを処理するハンドラ

    Parameters:
    -----------
    data: Dict[str, Any]
        StreamWriterから送信されたイベントデータ
    """
    # イベントをセッションに追加
    if "events" not in st.session_state:
        st.session_state.events = []

    st.session_state.events.append(data)

    # イベントタイプを取得
    event_type = data.get("event", "")

    # OCR完了イベント
    if event_type == EventType.OCR_DONE:
        st.session_state.ocr_text = data.get("text", "")
        st.session_state.ocr_result = data.get("structured_data", {})
        st.session_state.workflow_state = "ocr_complete"

    # 勘定科目提案完了イベント
    elif event_type == EventType.ACCOUNT_SUGGESTED:
        st.session_state.account_info = data.get("account_info", {})
        st.session_state.workflow_state = "account_suggested"

    # 保存完了イベント
    elif event_type == EventType.SAVE_COMPLETED:
        st.session_state.workflow_state = "complete"

    # エラーイベント
    elif event_type == EventType.ERROR:
        st.session_state.error_message = data.get("message", "エラーが発生しました")
        st.session_state.workflow_state = "error"


def process_interrupt_and_get_feedback(interrupt_data: Dict[str, Any]) -> None:
    """
    割り込みを処理してフィードバックを収集する

    Parameters:
    -----------
    interrupt_data: Dict[str, Any]
        割り込みデータ
    """
    # 状態を更新
    st.session_state.workflow_state = "feedback"

    # OCR結果と勘定科目情報を更新
    if "ocr_result" in interrupt_data:
        ocr_result = interrupt_data["ocr_result"]
        st.session_state.ocr_result = ocr_result

    if "account_info" in interrupt_data:
        account_info = interrupt_data["account_info"]

        # OCR結果から金額、インボイス番号などを追加
        if "ocr_result" in interrupt_data:
            ocr_result = interrupt_data["ocr_result"]

            # 金額を追加
            if "amount" in ocr_result and "amount" not in account_info:
                account_info["amount"] = ocr_result["amount"]

            # 消費税関連の情報
            if "other_info" in ocr_result:
                for info in ocr_result["other_info"]:
                    # 内税など税関連情報からインボイス関連データを設定
                    if (
                        info["key"] == "内税"
                        or info["key"] == "外税"
                        or info["key"] == "消費税"
                    ):
                        if "tax_amount" not in account_info:
                            # 税額が直接含まれていない場合は税率から計算（概算）
                            tax_rate = info["value"].replace("%", "")
                            try:
                                amount = float(ocr_result["amount"])
                                tax_rate = float(tax_rate) / 100
                                tax_amount = int(amount * tax_rate)
                                account_info["tax_amount"] = str(tax_amount)
                            except (ValueError, TypeError):
                                pass

                    # インボイス番号に関する情報
                    if info["key"] == "登録番号" or info["key"] == "インボイス番号":
                        account_info["invoice_number"] = info["value"]

        st.session_state.account_info = account_info
        # 元の状態は保存不要（編集不可のため）

    # フィードバックを表示・収集するために画面を更新
    st.rerun()


def handle_feedback_submission() -> Optional[Dict[str, Any]]:
    """
    ユーザーのフィードバックを収集して返す

    Returns:
    --------
    Optional[Dict[str, Any]]
        フィードバックデータ（コマンド形式）。ボタンが押されていない場合はNone
    """
    account_info = st.session_state.account_info

    # 勘定科目情報を表示
    account_info_editor(account_info)

    # アクションボタンを表示
    actions = display_action_buttons(False)  # 編集モードは常にFalse

    # ボタン押下時の処理
    if actions["approved"]:
        # 承認 - approveコマンドを返す
        print("承認ボタンが押されました")  # デバッグ出力
        return {"command": CommandType.APPROVE}

    elif actions["feedback"]:
        # フィードバックボタンが押された場合
        feedback_text = st.session_state.get("feedback_text", "")
        if feedback_text:
            # 直接LLMにフィードバックを送信して再生成を依頼
            print(f"フィードバックが送信されました: {feedback_text}")  # デバッグ出力
            return {"command": CommandType.REGENERATE, "feedback": feedback_text}

    # デフォルトは何も返さない
    return None


def process_workflow(
    input_data: Union[str, Dict[str, Any]], spinner_message: str = "処理中..."
) -> None:
    """
    ワークフローを処理する共通関数

    Parameters:
    -----------
    input_data: Union[str, Dict[str, Any]]
        処理する入力データ（画像パスまたはフィードバックデータ）
    spinner_message: str
        処理中に表示するメッセージ
    """
    # thread_idを設定
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        # 処理中の表示
        with st.spinner(spinner_message):
            # ワークフローの実行
            for chunk in receipt_workflow.stream(
                input_data,
                config=config,
            ):
                # デバッグ用
                print(chunk)

                # 中間結果の処理
                if isinstance(chunk, dict):
                    # OCR結果の処理
                    if "process_and_ocr_image" in chunk:
                        ocr_result = chunk["process_and_ocr_image"]
                        st.session_state.ocr_text = ocr_result.raw_text
                        st.session_state.ocr_result = {
                            "raw_text": ocr_result.raw_text,
                            "date": ocr_result.date,
                            "amount": ocr_result.amount,
                            "shop_name": ocr_result.shop_name,
                            "items": [
                                {"name": item.name, "price": item.price}
                                for item in ocr_result.items
                            ],
                            "other_info": [
                                {"key": info.key, "value": info.value}
                                for info in ocr_result.other_info
                            ],
                        }

                    # 勘定科目提案の処理
                    elif "generate_account_suggestion" in chunk:
                        account_info = chunk["generate_account_suggestion"]
                        account_data = {
                            "account": account_info.account,
                            "sub_account": account_info.sub_account,
                            "vendor": account_info.vendor,
                            "description": account_info.description,
                            "reason": account_info.reason,
                        }

                        # 追加フィールドがある場合は取得
                        if hasattr(account_info, "amount"):
                            account_data["amount"] = account_info.amount
                        if hasattr(account_info, "tax_amount"):
                            account_data["tax_amount"] = account_info.tax_amount
                        if hasattr(account_info, "invoice_number"):
                            account_data["invoice_number"] = account_info.invoice_number

                        # OCRから金額情報を取得（勘定科目提案に含まれていない場合）
                        if "amount" not in account_data and hasattr(
                            st.session_state, "ocr_result"
                        ):
                            ocr_result = st.session_state.ocr_result
                            if "amount" in ocr_result:
                                account_data["amount"] = ocr_result["amount"]

                        st.session_state.account_info = account_data

                    # 保存完了イベントの処理
                    elif "save_receipt" in chunk:
                        print("保存完了イベントを検出")
                        # 保存が完了したら完了状態に更新
                        st.session_state.workflow_state = "complete"
                        # 確実に再描画されるようにrerunを呼び出す
                        st.rerun()

                    # 割り込みの検出
                    elif "__interrupt__" in chunk:
                        interrupt = chunk["__interrupt__"][
                            0
                        ]  # Interruptオブジェクトを取得
                        interrupt_data = interrupt.value  # 割り込みデータを取得

                        # 割り込みを処理してフィードバックを表示
                        process_interrupt_and_get_feedback(interrupt_data)
                        return

                    # 完了イベントの処理
                    elif "receipt_workflow" in chunk:
                        workflow_data = chunk["receipt_workflow"]
                        print(f"ワークフローデータを検出: {workflow_data}")

                        # completed=Trueかチェック
                        if workflow_data.get("completed", False):
                            print("ワークフロー完了状態を検出")
                            st.session_state.workflow_state = "complete"

                            # OCR結果とアカウント情報を更新
                            if "ocr_result" in workflow_data:
                                st.session_state.ocr_result = workflow_data[
                                    "ocr_result"
                                ]
                            if "account_info" in workflow_data:
                                st.session_state.account_info = workflow_data[
                                    "account_info"
                                ]

                            # 確実に再描画されるようにrerunを呼び出す
                            st.rerun()

            # ループ終了後も状態を確認
            if st.session_state.workflow_state == "processing":
                print("完了イベントは検出されませんでしたが、処理は終了しました")
                st.session_state.workflow_state = "complete"
                st.rerun()

    except Exception as e:
        st.error(f"ワークフロー処理エラー: {str(e)}")
        st.session_state.workflow_state = "error"
        print(f"エラー詳細: {e}")


def start_workflow(image_path: str) -> None:
    """
    ワークフローを開始する

    Parameters:
    -----------
    image_path: str
        処理する画像のパス
    """
    # 状態のリセット
    st.session_state.workflow_state = "processing"

    # 一時ファイルを管理
    st.session_state.temp_files.append(image_path)

    # 共通処理関数を呼び出し
    process_workflow(image_path, "OCRとデータ処理中...")


def resume_workflow_with_feedback(feedback: Dict[str, Any]) -> None:
    """
    フィードバックを使ってワークフローを再開する

    Parameters:
    -----------
    feedback: Dict[str, Any]
        ユーザーのフィードバック
    """
    # 状態を更新
    st.session_state.workflow_state = "processing"

    # コマンドタイプをデバッグ出力
    command = feedback.get("command", "unknown")
    print(f"コマンドタイプ: {command}")

    # コマンドタイプによってスピナーメッセージを変更
    spinner_message = "処理中..."
    if feedback.get("command") == CommandType.REGENERATE:
        spinner_message = "フィードバックに基づいて再生成中..."
    elif feedback.get("command") == CommandType.APPROVE:
        spinner_message = "情報を保存中..."

    # Command.resumeを使ってワークフローを再開
    process_workflow(Command(resume=feedback), spinner_message)


def clean_up_temp_files() -> None:
    """一時ファイルをクリーンアップ"""
    for file_path in st.session_state.temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"一時ファイル削除エラー: {e}")

    # リストをクリア
    st.session_state.temp_files = []


def handle_on_change(updated_info: Dict[str, Any]) -> None:
    """
    フィールド値変更時のコールバック

    Parameters:
    -----------
    updated_info: Dict[str, Any]
        更新された勘定科目情報
    """
    # 変更内容をセッションに保存
    st.session_state.updated_account_info = updated_info


def main() -> None:
    """メインアプリケーション"""
    # ページ設定
    setup_page()

    # セッション状態の初期化
    init_session_state()

    # サイドバーメニュー
    with st.sidebar:
        st.subheader("メニュー")

        # 新規領収書処理ボタン - 処理中は無効化
        new_receipt_disabled = st.session_state.workflow_state in [
            "processing",
            "feedback",
        ]
        if st.button(
            "新規領収書処理", use_container_width=True, disabled=new_receipt_disabled
        ):
            st.session_state.display_mode = "input"
            # 状態をリセット
            st.session_state.workflow_state = "idle"
            st.session_state.ocr_text = ""
            st.session_state.ocr_result = {}
            st.session_state.account_info = {}
            # 一時ファイルのクリーンアップ
            clean_up_temp_files()
            st.rerun()

        # 履歴表示ボタン - 処理中は無効化
        history_disabled = st.session_state.workflow_state in ["processing", "feedback"]
        if st.button("履歴表示", use_container_width=True, disabled=history_disabled):
            st.session_state.display_mode = "history"
            st.rerun()

        # デバッグ情報
        if st.checkbox("デバッグ情報を表示", value=False):
            st.write(f"ワークフロー状態: {st.session_state.workflow_state}")
            st.write(f"スレッドID: {st.session_state.thread_id}")

    # 履歴表示モード
    if st.session_state.display_mode == "history":
        # 保存済みデータを取得して表示
        receipts = get_saved_receipts()
        display_receipt_history(receipts)
        return

    # 入力モード - メイン画面を左右に分割
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # 画像アップロード/カメラ入力部分
        image_path = handle_image_input()

        # 処理開始ボタン - 処理中は無効化
        start_disabled = not image_path or st.session_state.workflow_state != "idle"
        if image_path and st.session_state.workflow_state == "idle":
            if st.button(
                "処理開始",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
            ):
                # ワークフローを開始
                start_workflow(image_path)

        # 処理中の表示
        if st.session_state.workflow_state == "processing":
            display_loading_spinner("処理中...")

        # 現在の状態を表示（デバッグ用）
        st.caption(f"現在の状態: {st.session_state.workflow_state}")

    with right_col:
        # OCR結果の表示
        if st.session_state.ocr_text:
            display_ocr_text(st.session_state.ocr_text)

        # フィードバックUIの表示
        if st.session_state.workflow_state == "feedback":
            # ボタンクリック時にフィードバックデータを取得して次のステップへ
            feedback = handle_feedback_submission()
            if feedback:
                # フィードバックを使ってワークフローを再開
                resume_workflow_with_feedback(feedback)
                # この時点ではまだ処理中なので、rerunは不要

        # 処理完了後の表示
        if st.session_state.workflow_state == "complete":
            print("完了状態を検出: 成功メッセージを表示します")
            display_success_message()

            # バックグラウンドでCSVのバックアップを作成
            backup_csv()

        # エラー表示
        if st.session_state.workflow_state == "error" and hasattr(
            st.session_state, "error_message"
        ):
            st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()
