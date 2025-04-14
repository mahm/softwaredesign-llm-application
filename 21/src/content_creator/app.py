import uuid

import streamlit as st
from langgraph.types import Command

# src.からの絶対インポートを相対インポートに変更
from langraph_workflow import content_workflow
from ui_components import (
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
        st.session_state.workflow_state = "idle"  # idle, running, feedback
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = {}
    if "current_data" not in st.session_state:
        st.session_state.current_data = {}

    # サイドバーは最初に表示
    render_sidebar()

    # レイアウト - 2カラムレイアウト
    col1, col2 = st.columns([1, 1])

    with col1:
        # チャット欄
        st.subheader("チャット")

        # チャット履歴（LangGraphから取得）
        if "current_data" in st.session_state and st.session_state.current_data:
            render_messages(st.session_state.current_data.get("messages", []))

            # フィードバックオプション - チャット欄に表示
            if st.session_state.workflow_state == "feedback":
                feedback_options = st.session_state.current_data.get("options", [])

                # フィードバックの処理
                feedback = render_feedback_options(feedback_options)

                # フィードバックが提供された場合
                if feedback:
                    # ワークフロー状態を実行中に設定
                    st.session_state.workflow_state = "running"

                    # 処理中メッセージ
                    with st.spinner("フィードバックを処理中..."):
                        try:
                            # ワークフローを再開
                            process_workflow(feedback)
                        except Exception as e:
                            st.session_state.debug_info["feedback_error"] = str(e)
                            st.error(f"エラーが発生しました: {e}")
                            # エラー発生時は入力可能に戻す
                            st.session_state.workflow_state = "idle"
                            st.rerun()

        # フィードバック待ちでない場合のみ入力フィールドを表示
        if st.session_state.workflow_state != "feedback":
            # 新しい指示の入力
            prompt = render_chat_input()

            if prompt:
                # デバッグ情報
                st.session_state.debug_info["last_prompt"] = prompt

                # ワークフロー状態を実行中に設定
                st.session_state.workflow_state = "running"

                # last_feedbackをリセット
                st.session_state.last_feedback = None

                # 処理中メッセージ
                with st.spinner("コンテンツを生成中..."):
                    # ワークフロー実行 - 新しいプロンプトを入力
                    process_workflow(prompt)

    # 右カラム - コンテンツ表示
    with col2:
        # 成果物表示欄
        st.subheader("生成されたコンテンツ")

        # コンテンツ表示
        if "current_data" in st.session_state and st.session_state.current_data:
            content = st.session_state.current_data.get("content", "")

            if content:
                # コンテンツ表示
                render_content_area(content)

                # ステータス情報（デバッグ用）
                with st.expander("デバッグ情報", expanded=False):
                    st.write(f"ステータス: {st.session_state.workflow_state}")
                    if st.session_state.debug_info:
                        st.json(st.session_state.debug_info)
        else:
            st.info("チャット欄に指示を入力すると、ここにコンテンツが表示されます。")


def process_workflow(user_input):
    """
    ワークフローを実行し、結果またはinterruptを処理する

    Args:
        input_data: 新しいプロンプト辞書、またはCommand(resume=feedback)
    """
    # ユーザー入力を保存
    input_data = {"user_input": user_input}

    # LangGraphで使用するスレッドID設定
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # ワークフロー実行
    for chunk in content_workflow.stream(
        input=input_data,
        config=config,
    ):
        print("chunk", chunk)
        if "content_workflow" in chunk:
            # デバッグ情報にinterrupt_dataを保存
            content_workflow_data = chunk["content_workflow"]
            st.session_state.debug_info["interrupt_data"] = content_workflow_data

            # ワークフロー状態をフィードバック待ちに設定
            st.session_state.workflow_state = "feedback"

            # 抽出したデータをUIで利用できるように保存
            st.session_state.current_data = content_workflow_data

    # 全処理完了後にrerun()を実行
    st.rerun()


if __name__ == "__main__":
    main()
