from typing import Any, Dict, List

import streamlit as st


def setup_page_config():
    """Streamlitページ設定"""
    st.set_page_config(
        page_title="Content Creator",
        page_icon="📝",
        layout="wide",
    )

    # 全体的なパネルスタイルのCSSを定義
    st.markdown(
        """
        <style>
        .panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            padding: 1.5rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .panel-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #dee2e6;
        }
        .feedback-panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            padding: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .content-box {
            background-color: transparent;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            padding: 1.5rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            height: 40vh;
            overflow-y: auto;
            font-family: 'Source Sans Pro', sans-serif;
        }
        .content-box h1, .content-box h2, .content-box h3 {
            color: #212529;
            margin-bottom: 1rem;
        }
        .character-count {
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """サイドバーのレンダリング"""
    with st.sidebar:
        st.title("Content Creator")
        st.markdown(
            "このアプリは、LangGraph Functional APIを使用して、ユーザーの指示に基づいてコンテンツを生成します。"
        )
        st.markdown("---")
        st.markdown("### 使い方")
        st.markdown("1. 左側のチャット欄に指示を入力します")
        st.markdown("2. 右側に生成されたコンテンツが表示されます")
        st.markdown("3. フィードバックを選択するか自由に入力して改善できます")

        # ステータス表示
        st.markdown("---")
        st.markdown("### ステータス")
        if "workflow_state" in st.session_state:
            if st.session_state.workflow_state == "idle":
                st.success("準備完了")
            elif st.session_state.workflow_state == "feedback":
                st.warning("フィードバック待ち")


def render_chat_input():
    """チャット入力エリアのレンダリング"""
    prompt = st.chat_input("指示を入力してください...")
    return prompt


def render_messages(messages: List[Dict[str, Any]]):
    """チャット履歴のレンダリング"""
    for message in messages:
        try:
            role = message["role"]
            content = message["content"]

            # チャットメッセージを表示
            with st.chat_message(role):
                st.write(content)
        except Exception as e:
            # メッセージ表示に失敗した場合は警告
            with st.chat_message("system"):
                st.warning(f"メッセージの表示に失敗しました: {str(e)}")


def render_content_area(content: str):
    """コンテンツ表示エリアのレンダリング - 右カラムでの表示に最適化"""
    with st.container():
        # コンテンツが存在する場合は文字数を表示
        if content:
            # HTMLタグを除いた純粋なテキストの文字数をカウント
            char_count = len(content)
            st.markdown(
                f'<div class="character-count">{char_count}文字</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="content-box">{content}</div>', unsafe_allow_html=True)


def render_feedback_options(options: List[str]):
    """フィードバックオプションのレンダリング"""
    with st.container(border=True):
        opt_cols = st.columns(3)
        for i, option in enumerate(options[:3]):  # 最大3つまで表示
            with opt_cols[i]:
                if st.button(
                    option.strip(),
                    key=f"feedback_opt_{i}",
                    type="primary" if i == 0 else "secondary",
                    use_container_width=True,
                ):
                    return option

        # 自由入力フィールド
        st.caption("または、自由にフィードバックを入力:")

        # 送信後にフィールドをクリアするための状態管理
        if "custom_feedback_submitted" not in st.session_state:
            st.session_state.custom_feedback_submitted = False

        if st.session_state.custom_feedback_submitted:
            st.session_state.custom_feedback = ""
            st.session_state.custom_feedback_submitted = False

        custom_feedback = st.text_input(
            "フィードバック", key="custom_feedback", label_visibility="collapsed"
        )

        # Enterキーが押された場合の処理
        if custom_feedback:
            st.session_state.custom_feedback_submitted = True
            return custom_feedback

        # 送信ボタン
        if st.button("送信", key="submit_feedback"):
            if custom_feedback:
                st.session_state.custom_feedback_submitted = True
                return custom_feedback
