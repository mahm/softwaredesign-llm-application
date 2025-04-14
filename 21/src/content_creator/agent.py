from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from pydantic import BaseModel

# LLMの初期化 - Claude 3.7 Sonnet
llm = ChatAnthropic(
    model_name="claude-3-7-sonnet-20250219",
    temperature=0.7,
    timeout=60,
    stop=None,
)


@task
def generate_content(
    theme: str,
    messages: Optional[List[BaseMessage]] = None,
) -> str:
    """指定されたユーザー入力とメッセージに基づいてコンテンツを生成する"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはX（旧：Twitter）のポストを作成するエージェントです。ユーザーから与えられたテーマに基づいて、Xのポストを作成してください。曖昧な指示であっても、仮説ベースでコンテンツを作成するようにしてください。",
            ),
            (
                "user",
                "次のテーマにしたがってポストを作成してください：{theme}{feedback_context}\n\nXに投稿するポストのみを出力すること。",
            ),
        ]
    )

    # フィードバック履歴があれば追加
    feedback_context = ""
    if messages and len(messages) > 0:
        feedback_context = "\n\n以下はこれまでに受け取ったフィードバックです。これらすべてを考慮して改善してください：\n"
        for i, message in enumerate(messages, 1):
            # HumanMessageのみを抽出
            if isinstance(message, HumanMessage):
                feedback_context += f"{i}. {message.content}\n"
            elif (
                isinstance(message, Dict)
                and "role" in message
                and message["role"] == "user"
            ):
                feedback_context += f"{i}. {message['content']}\n"

    # LLMを呼び出してコンテンツを生成
    chain = prompt | llm
    response = chain.invoke(
        {
            "theme": theme,
            "feedback_context": feedback_context,
        }
    )
    return str(response.content)


class FeedbackList(BaseModel):
    values: List[str]


@task
def generate_feedback_options(content: str) -> List[str]:
    """コンテンツに対するフィードバック候補を生成する"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはコンテンツをより魅力的なものにするためのフィードバック候補を生成するアシスタントです。",
            ),
            (
                "user",
                "以下のX（旧：Twitter）向けのポストに対する3つの異なるフィードバック候補を生成してください。それぞれ異なる観点から改善点を指摘してください。\n\n重要: 各フィードバックは20文字以内の簡潔な指示文にしてください。ボタンラベルとしても使用できる長さが必要です。\n\nコンテンツ:\n{content}",
            ),
        ]
    )
    chain = prompt | llm.with_structured_output(FeedbackList)
    response: FeedbackList = chain.invoke({"content": content})  # type: ignore

    return response.values


@entrypoint(checkpointer=MemorySaver())
def workflow(
    inputs: Dict[str, Any],
    *,
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # 以前の状態を取得または初期化
    state = previous or {
        "theme": "",
        "content": "",
        "options": [],
        "messages": [],
    }

    # 最初の指示をコンテンツのテーマとして設定する
    if state["theme"] == "":
        state["theme"] = inputs["user_input"]

    # ユーザーの指示をメッセージに追加
    state["messages"].append({"role": "user", "content": inputs["user_input"]})

    # コンテンツを生成
    content = generate_content(state["theme"], state["messages"]).result()
    state["content"] = content

    # フィードバックオプションを生成
    feedback_options = generate_feedback_options(content).result()
    state["options"] = feedback_options

    # アシスタントメッセージを追加
    message = "コンテンツを生成しました。フィードバックをお願いします。"
    state["messages"].append({"role": "assistant", "content": message})

    # 最終的なステートを返し、チェックポイントに保存
    return state
