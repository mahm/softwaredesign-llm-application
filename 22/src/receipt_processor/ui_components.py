"""
Streamlit UI関連コンポーネント
"""

import tempfile
import time
from typing import Optional

import pandas as pd
import streamlit as st

from src.receipt_processor.models import AccountInfo, CommandType, Feedback


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


def account_info_editor(account_info: AccountInfo) -> None:
    """
    勘定科目情報の表示

    Parameters:
    -----------
    account_info: Dict[str, Any]
        表示する勘定科目情報

    Returns:
    --------
    Dict[str, Any]
        元の勘定科目情報
    """
    # 生成理由
    st.markdown(account_info.reason)

    st.subheader("勘定科目情報")

    # 金額関連の情報（2列レイアウト）
    col1, col2 = st.columns(2)

    with col1:
        # 勘定科目
        st.text_input(
            "勘定科目",
            value=account_info.account,
            key="account_input",
            disabled=True,
        )

        # 取引先
        st.text_input(
            "取引先",
            value=account_info.vendor,
            key="vendor_input",
            disabled=True,
        )

        # 金額
        st.text_input(
            "金額",
            value=account_info.amount,
            key="amount_input",
            disabled=True,
        )

    with col2:
        # 補助科目
        st.text_input(
            "補助科目",
            value=account_info.sub_account,
            key="sub_account_input",
            disabled=True,
        )

        # インボイス番号
        st.text_input(
            "インボイス番号",
            value=account_info.invoice_number,
            key="invoice_number_input",
            disabled=True,
        )

        # 消費税額
        st.text_input(
            "消費税額",
            value=account_info.tax_amount,
            key="tax_amount_input",
            disabled=True,
        )

    # 摘要
    st.text_area(
        "摘要",
        value=account_info.description,
        key="description_input",
        disabled=True,
    )


def display_action_buttons() -> Optional[Feedback]:
    """
    アクションボタンを表示

    Returns:
    --------
    Optional[Feedback]
        フィードバック。ボタンが押されなかった場合はNone
    """
    st.subheader("アクション")

    # 承認ボタン
    if st.button(
        "この情報で承認する",
        type="primary",
        use_container_width=True,
    ):
        feedback = Feedback(
            command=CommandType.APPROVE,
            content="",
        )
        return feedback

    # 罫線（区切り線）
    st.markdown("---")

    # 単純なform外テキストエリアとボタンの組み合わせ
    feedback_text = st.text_area(
        "フィードバック内容",
        placeholder="例：「補助科目を交通費から駐車場代に変更して」「取引先名をカタカナで表記して」「摘要に日付を含めて」など",
        key="direct_feedback",
        height=100,
    )

    # 説明文
    st.markdown(
        "自然言語でエージェントにフィードバックを送り、勘定科目情報を再生成できます"
    )

    # 送信ボタン
    if st.button(
        "フィードバックを送信して再生成",
        type="secondary",
        use_container_width=True,
        key="send_feedback",
    ):
        if feedback_text:
            feedback = Feedback(
                command=CommandType.REGENERATE,
                content=feedback_text,
            )
            return feedback

    return None


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
