import uuid

import streamlit as st

from content_creator.agent import workflow
from content_creator.ui_components import (
    render_chat_input,
    render_content_area,
    render_feedback_options,
    render_messages,
    render_sidebar,
    setup_page_config,
)


def main():
    """メインアプリケーション関数"""
    # ページ設定
    setup_page_config()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = "idle"  # idle, feedback
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = {}
    if "current_data" not in st.session_state:
        st.session_state.current_data = {}

    # サイドバーは最初に表示
    render_sidebar()

    # レイアウト - 2カラムレイアウト
    col1, col2 = st.columns([1, 1])

    # 左カラム - チャット欄
    with col1:
        st.subheader("チャット")

        # チャット履歴（LangGraphから取得）
        if "current_data" in st.session_state and st.session_state.current_data:
            render_messages(st.session_state.current_data.get("messages", []))

            # フィードバックオプション - チャット欄に表示
            if st.session_state.workflow_state == "feedback":
                # フィードバックの選択肢を取得
                feedback_options = st.session_state.current_data.get("options", [])

                # フィードバック入力欄の表示
                feedback = render_feedback_options(feedback_options)

                # フィードバックが提供された場合
                if feedback:
                    with st.spinner("フィードバックを反映中..."):
                        # ワークフローを再開
                        process_workflow(feedback)

        # フィードバック待ちでない場合のみ入力フィールドを表示
        if st.session_state.workflow_state != "feedback":
            # 新しい指示の入力
            prompt = render_chat_input()

            # ユーザーからの指示が入力された場合
            if prompt:
                with st.spinner("コンテンツを生成中..."):
                    # ワークフロー実行 - 新しいプロンプトを入力
                    process_workflow(prompt)

    # 右カラム - コンテンツ表示
    with col2:
        st.subheader("生成されたコンテンツ")

        # コンテンツ表示
        if "current_data" in st.session_state and st.session_state.current_data:
            content = st.session_state.current_data.get("content", "")

            if content:
                # コンテンツの内容をレンダリング
                render_content_area(content)

                # ステータス情報（デバッグ用）
                with st.expander("デバッグ情報", expanded=False):
                    st.write(f"ステータス: {st.session_state.workflow_state}")
                    if st.session_state.debug_info:
                        st.json(st.session_state.debug_info)
        else:
            st.info("チャット欄に指示を入力すると、ここにコンテンツが表示されます。")


def process_workflow(user_input: str):
    """
    ワークフローを実行する

    Args:
        user_input: ユーザーからの入力
    """
    # ユーザー入力
    input_data = {"user_input": user_input}

    # スレッドIDの設定
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # ワークフロー実行
    for chunk in workflow.stream(
        input=input_data,
        config=config,
    ):
        if "workflow" in chunk:
            # デバッグ情報にinterrupt_dataを保存
            workflow_data = chunk["workflow"]
            st.session_state.debug_info["interrupt_data"] = workflow_data

            # ワークフロー状態をフィードバック待ちに設定
            st.session_state.workflow_state = "feedback"

            # 抽出したデータをUIで利用できるように保存
            st.session_state.current_data = workflow_data

    # 全処理完了後にrerun()を実行
    st.rerun()


if __name__ == "__main__":
    main()
