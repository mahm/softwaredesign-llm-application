from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from pydantic import BaseModel

# チェックポインターの初期化 - メモリ内保存
checkpointer = MemorySaver()

# LLMの初期化 - Claude
# Claude 3 Sonnetのモデル名を正しく設定
llm = ChatAnthropic(
    model_name="claude-3-sonnet-20240229", temperature=0, timeout=60, stop=None
)


@task
def generate_content(
    user_input: str,
    messages: Optional[List[BaseMessage]] = None,
) -> str:
    """指定されたユーザー入力とメッセージに基づいてコンテンツを生成する"""
    prompt_messages: List[BaseMessage] = []

    # システムプロンプト
    system_message = "あなたは高品質なコンテンツを作成するアシスタントです。ユーザーの指示に基づいて、明確で構造化された成果物を作成してください。曖昧な指示であっても、仮説ベースでコンテンツを作成するようにしてください。"
    prompt_messages.append(SystemMessage(content=system_message))

    # 基本プロンプト
    base_prompt = f"以下の指示に従って成果物を作成してください：\n{user_input}"

    # フィードバック履歴があれば追加
    if messages and len(messages) > 0:
        feedback_context = "\n\n以下はこれまでに受け取ったフィードバックです。これらすべてを考慮して改善してください：\n"
        for i, message in enumerate(messages, 1):
            # HumanMessageのみを抽出
            if isinstance(message, HumanMessage):
                feedback_context += f"{i}. {message.content}\n"
        base_prompt += feedback_context

    # HumanMessageをリストに追加
    prompt_messages.append(HumanMessage(content=base_prompt))

    # LLMを呼び出してコンテンツを生成
    response = llm.invoke(prompt_messages)
    return str(response.content)


class FeedbackList(BaseModel):
    values: List[str]


@task
def generate_feedback_options(content: str) -> List[str]:
    """コンテンツに対するフィードバック候補を生成する"""
    messages = [
        SystemMessage(
            content="あなたは簡潔で的確なフィードバックを提供するアシスタントです。"
        ),
        HumanMessage(
            content=f"以下のコンテンツに対する3つの異なるフィードバック候補を生成してください。それぞれ異なる観点から改善点を指摘してください。\n\n重要: 各フィードバックは20文字以内の簡潔な指示文にしてください。ボタンラベルとしても使用できる長さが必要です。\n\nコンテンツ:\n{content}"
        ),
    ]

    chain = llm.with_structured_output(FeedbackList)
    response: FeedbackList = chain.invoke(messages)  # type: ignore

    return response.values


@entrypoint(checkpointer=checkpointer)
def content_workflow(
    inputs: Dict[str, Any],
    *,
    previous: Optional[Dict[str, Any]] = None,
) -> entrypoint.final[Dict[str, Any], Dict[str, Any]]:
    # 以前の状態を取得または初期化
    state = previous or {
        "user_input": "",
        "content": "",
        "iteration": 0,
        "messages": [],
    }

    state["user_input"] = inputs["user_input"]
    state["messages"].append({"role": "user", "content": state["user_input"]})

    # 反復回数をインクリメント
    state["iteration"] += 1

    # コンテンツを生成
    content = generate_content(state["user_input"], state["messages"]).result()
    state["content"] = content

    # フィードバックオプションを動的に生成
    feedback_options = generate_feedback_options(content).result()

    # アシスタントメッセージを追加
    message = "コンテンツを生成しました。フィードバックをお願いします。"
    state["messages"].append({"role": "assistant", "content": message})
    print("state", state)

    # UIに表示する値と、チェックポイントに保存する値を分離
    return entrypoint.final(
        value={  # UIに返す値
            "content": state.get("content", ""),
            "options": feedback_options,
            "messages": state.get("messages", []),
        },
        save=state,  # 完全な状態をチェックポイントに保存
    )
