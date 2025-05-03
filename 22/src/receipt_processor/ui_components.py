"""
Streamlit UI関連コンポーネント
"""

import io
import tempfile
import time
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


def setup_page() -> None:
    """ページの基本設定"""
    st.set_page_config(
        page_title="領収書OCRエージェント",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("領収書OCRエージェント")


def handle_image_input() -> Optional[str]:
    """
    Streamlitでの画像入力処理（アップロードのみ）

    Returns:
    --------
    Optional[str]
        一時保存した画像ファイルのパス。画像がアップロードされていない場合はNone
    """
    st.subheader("領収書画像を選択")

    # ファイルアップロード機能
    uploaded_file = st.file_uploader(
        "領収書画像をアップロード", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        # 画像プレビュー表示
        st.image(uploaded_file, caption="アップロード画像", use_container_width=True)

        # 処理用の一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name
        return image_path

    return None


def display_ocr_text(ocr_text: str) -> None:
    """OCRテキストを表示"""
    with st.expander("OCR抽出テキスト", expanded=False):
        st.text_area("抽出されたテキスト", ocr_text, height=200, disabled=True)


def account_info_editor(
    account_info: Dict[str, Any], on_change: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    勘定科目情報の表示

    Parameters:
    -----------
    account_info: Dict[str, Any]
        表示する勘定科目情報
    on_change: Optional[Callable]
        値が変更されたときに呼び出すコールバック関数（現在は使用しない）

    Returns:
    --------
    Dict[str, Any]
        元の勘定科目情報
    """
    # 生成理由（常に表示）
    st.markdown(account_info.get("reason", ""))

    st.subheader("勘定科目情報")

    # 金額関連の情報（2列レイアウト）
    col1, col2 = st.columns(2)

    with col1:
        # 勘定科目
        st.text_input(
            "勘定科目",
            value=account_info.get("account", ""),
            key="account_input",
            disabled=True,
        )

        # 取引先
        st.text_input(
            "取引先",
            value=account_info.get("vendor", ""),
            key="vendor_input",
            disabled=True,
        )

        # 金額
        st.text_input(
            "金額",
            value=account_info.get("amount", ""),
            key="amount_input",
            disabled=True,
        )

    with col2:
        # 補助科目
        st.text_input(
            "補助科目",
            value=account_info.get("sub_account", ""),
            key="sub_account_input",
            disabled=True,
        )

        # インボイス番号
        st.text_input(
            "インボイス番号",
            value=account_info.get("invoice_number", ""),
            key="invoice_number_input",
            disabled=True,
        )

        # 消費税額
        st.text_input(
            "消費税額",
            value=account_info.get("tax_amount", ""),
            key="tax_amount_input",
            disabled=True,
        )

    # 摘要
    st.text_area(
        "摘要",
        value=account_info.get("description", ""),
        key="description_input",
        disabled=True,
    )

    # 元の情報をそのまま返す
    return account_info


def display_action_buttons(is_modified: bool) -> Dict[str, bool]:
    """
    アクションボタンを表示

    Parameters:
    -----------
    is_modified: bool
        フィールドが変更されたかどうか（現在は使用しない）

    Returns:
    --------
    Dict[str, bool]
        ボタン押下状態
    """
    st.subheader("アクション")

    actions = {
        "approved": False,
        "feedback": False,
    }

    # 承認ボタン - 常に表示
    if st.button("この情報で承認する", type="primary", use_container_width=True):
        actions["approved"] = True

    # 罫線（区切り線）
    st.markdown("---")

    # フィードバックボタンが押された時のコールバック
    def on_feedback_submit() -> None:
        # 送信後にテキストエリアをクリア
        st.session_state.direct_feedback = ""
        # フィードバックを送信するフラグを設定
        st.session_state.submit_feedback = True

    # フィードバック入力欄
    feedback = st.text_area(
        "フィードバック内容",
        placeholder="例：「補助科目を交通費から駐車場代に変更して」「取引先名をカタカナで表記して」「摘要に日付を含めて」など",
        key="direct_feedback",
        height=100,
    )
    st.markdown(
        "自然言語でエージェントにフィードバックを送り、勘定科目情報を再生成できます"
    )

    # フィードバックボタン
    if st.button(
        "フィードバックを送信して再生成",
        type="secondary",
        use_container_width=True,
        on_click=on_feedback_submit if feedback else None,
    ):
        if feedback:
            actions["feedback"] = True
            # セッションにフィードバックを保存
            st.session_state.feedback_text = feedback

    # 前の実装との互換性のためのチェック（セッション状態から取得）
    if "submit_feedback" in st.session_state and st.session_state.submit_feedback:
        actions["feedback"] = True
        st.session_state.feedback_text = st.session_state.get(
            "direct_feedback_before_clear", ""
        )
        st.session_state.submit_feedback = False

    # 送信前にフィードバックの内容を保存（クリアされる前に保存）
    if feedback:
        st.session_state.direct_feedback_before_clear = feedback

    return actions


def display_receipt_history(receipts: list) -> None:
    """
    保存された領収書履歴を表示

    Parameters:
    -----------
    receipts: list
        領収書データのリスト
    """
    if not receipts:
        st.info("登録された領収書はありません")
        return

    st.subheader("登録済み領収書一覧")

    # DataFrameに変換して表示
    df = pd.DataFrame(receipts)

    # 生テキストは表示から除外
    if "raw_text" in df.columns:
        df = df.drop(columns=["raw_text"])

    # データフレームを表示
    st.dataframe(df, use_container_width=True)

    # CSVダウンロード機能
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name="receipts_data.csv",
        mime="text/csv",
    )


def display_success_message() -> None:
    """保存成功メッセージを表示"""
    st.success("領収書データが正常に保存されました！", icon="✅")
    print("成功メッセージを表示します")  # デバッグ出力

    # 次のアクションを促すメッセージ
    st.subheader("次のアクション")

    # 新しい領収書を処理するボタン
    if st.button(
        "新しい領収書を登録する",
        type="primary",
        use_container_width=True,
        key="new_receipt_btn",
    ):
        # セッション状態をリセット
        st.session_state.workflow_state = "idle"
        st.session_state.ocr_text = ""
        st.session_state.ocr_result = {}
        st.session_state.account_info = {}
        # 画面を再読み込み
        st.rerun()


def display_loading_spinner(message: str = "処理中...") -> None:
    """ローディングスピナーを表示"""
    with st.spinner(message):
        time.sleep(0.1)  # スピナー表示のための最小遅延
        # 実際の処理は呼び出し側で行うため、ここでは何もしない
