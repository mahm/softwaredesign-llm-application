"""
Streamlitアプリケーション
"""

import os
import uuid
from typing import Any, Dict, Optional, Union

import streamlit as st
from langgraph.types import Command

from src.receipt_processor.agent import receipt_workflow
from src.receipt_processor.models import (
    AccountInfo,
    CommandType,
    DisplayMode,
    EventType,
    Feedback,
    ReceiptOCRResult,
    WorkflowState,
)
from src.receipt_processor.storage import get_saved_receipts
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


def init_session_state(force: bool = False) -> None:
    """
    セッション状態の初期化

    Parameters:
    -----------
    force: bool, default=False
        Trueの場合、既存のセッション状態を上書きして初期化します
    """
    # スレッドIDの初期化
    if force or "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # ワークフロー状態の初期化
    if force or "workflow_state" not in st.session_state:
        st.session_state.workflow_state = WorkflowState.IDLE

    # OCRデータの初期化
    if force or "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if force or "ocr_result" not in st.session_state:
        st.session_state.ocr_result = {}

    # 会計情報の初期化
    if force or "account_info" not in st.session_state:
        st.session_state.account_info = {}

    # 一時ファイル管理
    if force or "temp_files" not in st.session_state:
        st.session_state.temp_files = []

    # 表示モード
    if force or "display_mode" not in st.session_state:
        st.session_state.display_mode = DisplayMode.INPUT


def handle_feedback_submission() -> Optional[Feedback]:
    """
    ユーザーのフィードバックを収集して返す

    Returns:
    --------
    Optional[Feedback]
        フィードバック。ボタンが押されなかった場合はNone
    """
    account_info = st.session_state.account_info

    # 勘定科目情報を表示
    account_info_editor(account_info)

    # アクションボタンを表示
    feedback: Optional[Feedback] = display_action_buttons()

    return feedback


def process_workflow(
    input_data: Union[str, Command],
    spinner_message: str = "処理中...",
) -> None:
    """
    ワークフローを処理する共通関数

    Parameters:
    -----------
    input_data: Union[str, Command]
        処理する入力データ（画像パスまたはフィードバックデータ）
    spinner_message: str
        処理中に表示するメッセージ
    """
    # thread_idを設定
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # 処理中の表示
    with st.spinner(spinner_message):
        try:
            # ワークフローの実行
            stream_iterator = receipt_workflow.stream(
                input_data, config=config, stream_mode=["custom", "values"]
            )

            for mode, payload in stream_iterator:
                # ペイロードがNoneの場合はスキップ
                if payload is None:
                    continue

                if mode == "custom":
                    event = payload.get("event", "")

                    if event == EventType.OCR_DONE:
                        raw_text = payload["text"]
                        ocr_result = ReceiptOCRResult.model_validate(
                            payload["structured_data"]
                        )
                        st.session_state.ocr_text = raw_text
                        st.session_state.ocr_result = ocr_result

                    elif event == EventType.ACCOUNT_SUGGESTED:
                        account_info = AccountInfo.model_validate(
                            payload["account_info"]
                        )
                        st.session_state.account_info = account_info

                    elif event == EventType.SAVE_COMPLETED:
                        st.session_state.workflow_state = (
                            WorkflowState.WORKFLOW_COMPLETED
                        )
                        st.rerun()

                    else:
                        st.session_state.error_message = (
                            f"未知のイベントを受信しました: {event}"
                        )
                        st.session_state.workflow_state = WorkflowState.ERROR
                        st.rerun()

                elif mode == "values":
                    # 割り込みの検出
                    if "__interrupt__" in payload:
                        interrupt = payload["__interrupt__"][0]

                        ocr_result = ReceiptOCRResult.model_validate(
                            interrupt.value.get("ocr_result")
                        )
                        account_info = AccountInfo.model_validate(
                            interrupt.value.get("account_info")
                        )
                        st.session_state.ocr_result = ocr_result
                        st.session_state.account_info = account_info
                        st.session_state.workflow_state = WorkflowState.WAIT_FEEDBACK
                        st.rerun()

        except Exception as e:
            print(f"ワークフロー実行エラー: {e}")
            import traceback

            print(f"詳細エラー: {traceback.format_exc()}")
            st.session_state.error_message = (
                f"ワークフロー実行中にエラーが発生しました: {e}"
            )
            st.session_state.workflow_state = WorkflowState.ERROR
            st.rerun()


def start_workflow(image_path: str) -> None:
    """
    ワークフローを開始する

    Parameters:
    -----------
    image_path: str
        処理する画像のパス
    """
    # 状態のリセット
    st.session_state.workflow_state = WorkflowState.PROCESSING

    # 一時ファイルを管理
    st.session_state.temp_files.append(image_path)

    # 共通処理関数を呼び出し
    process_workflow(image_path, "OCRとデータ処理中...")


def resume_workflow_with_feedback(feedback: Feedback) -> None:
    """
    フィードバックを使ってワークフローを再開する

    Parameters:
    -----------
    feedback: Feedback
        ユーザーのフィードバック
    """
    # 状態を更新
    st.session_state.workflow_state = WorkflowState.PROCESSING

    # コマンドタイプによってスピナーメッセージを変更
    spinner_message = "処理中..."
    if feedback.command == CommandType.REGENERATE:
        spinner_message = "フィードバックに基づいて再生成中..."
    elif feedback.command == CommandType.APPROVE:
        spinner_message = "情報を保存中..."

    # ワークフローを再開
    process_workflow(
        Command(resume=feedback.model_dump()),
        spinner_message,
    )


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
            WorkflowState.PROCESSING,
            WorkflowState.WAIT_FEEDBACK,
        ]
        if st.button(
            "新規領収書処理", use_container_width=True, disabled=new_receipt_disabled
        ):
            init_session_state(force=True)
            clean_up_temp_files()  # 一時ファイルをクリーンアップ
            st.rerun()

        # 履歴表示ボタン - 処理中は無効化
        history_disabled = st.session_state.workflow_state in [
            WorkflowState.PROCESSING,
            WorkflowState.WAIT_FEEDBACK,
        ]
        if st.button("履歴表示", use_container_width=True, disabled=history_disabled):
            st.session_state.display_mode = DisplayMode.HISTORY
            st.rerun()

        # デバッグ情報
        if st.checkbox("デバッグ情報を表示", value=False):
            st.write(f"ワークフロー状態: {st.session_state.workflow_state}")
            st.write(f"スレッドID: {st.session_state.thread_id}")

    # 履歴表示モード
    if st.session_state.display_mode == DisplayMode.HISTORY:
        # 保存済みデータを取得して表示
        receipts = get_saved_receipts()
        display_receipt_history(receipts)
        return

    # 入力モード - メイン画面を左右に分割
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # 画像アップロード
        image_path = handle_image_input()

        # 処理開始ボタン - 処理中は無効化
        start_disabled = (
            not image_path or st.session_state.workflow_state != WorkflowState.IDLE
        )
        if image_path and st.session_state.workflow_state == WorkflowState.IDLE:
            if st.button(
                "処理開始",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
            ):
                # ワークフローを開始
                start_workflow(image_path)

        # 処理中の表示
        if st.session_state.workflow_state == WorkflowState.PROCESSING:
            display_loading_spinner("処理中...")

        # 現在の状態を表示（デバッグ用）
        st.caption(f"現在の状態: {st.session_state.workflow_state}")

    with right_col:
        # OCR結果の表示
        if st.session_state.ocr_text:
            display_ocr_text(st.session_state.ocr_text)

        # フィードバックUIの表示
        if st.session_state.workflow_state == WorkflowState.WAIT_FEEDBACK:
            # ボタンクリック時にフィードバックデータを取得して次のステップへ
            feedback: Optional[Feedback] = handle_feedback_submission()
            if feedback:
                resume_workflow_with_feedback(feedback)

        # 処理完了後の表示
        if st.session_state.workflow_state == WorkflowState.WORKFLOW_COMPLETED:
            display_success_message()

        # エラー表示
        if st.session_state.workflow_state == WorkflowState.ERROR and hasattr(
            st.session_state, "error_message"
        ):
            st.error(st.session_state.error_message)


if __name__ == "__main__":
    main()
