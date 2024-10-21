from typing import Any, Literal
from uuid import uuid4

import streamlit as st
from agent import HumanInTheLoopAgent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def show_message(type: Literal["human", "agent"], title: str, message: str) -> None:
    with st.chat_message(type):
        st.markdown(f"**{title}**")
        st.markdown(message)


def app() -> None:
    load_dotenv(override=True)

    st.title("Human-in-the-loopを適用したリサーチエージェント")

    # st.session_stateにagentを保存
    if "agent" not in st.session_state:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        _agent = HumanInTheLoopAgent(_llm)
        _agent.subscribe(show_message)
        st.session_state.agent = _agent

    agent = st.session_state.agent

    # グラフを表示
    with st.sidebar:
        st.image(agent.mermaid_png())

    # st.session_stateにthread_idを保存
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    st.write(f"thread_id: {thread_id}")

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    if human_message:
        with st.spinner():
            agent.handle_human_message(human_message, thread_id)
        # ユーザー入力があった場合、承認状態をリセット
        st.session_state.approval_state = "pending"

    # 次がhuman_approvalの場合は承認ボタンを表示
    if agent.is_next_human_approval_node(thread_id):
        if "approval_state" not in st.session_state:
            st.session_state.approval_state = "pending"

        if st.session_state.approval_state == "pending":
            approved = st.button("承認")
            if approved:
                st.session_state.approval_state = "processing"
                st.rerun()
        elif st.session_state.approval_state == "processing":
            with st.spinner("タスク処理中..."):
                agent.handle_human_message("[APPROVE]", thread_id)
            st.session_state.approval_state = "pending"


app()
