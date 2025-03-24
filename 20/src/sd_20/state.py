from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import Messages, add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from typing_extensions import TypedDict

MAX_TOKENS = 128_000

def add_and_trim_messages(
    left_messages: Messages,
    right_messages: Messages
) -> Messages:
    """
    メッセージを結合した後、指定されたトークン数に基づいてトリムします。

    Args:
        left_messages: ベースとなるメッセージリスト
        right_messages: 追加するメッセージリスト
        max_tokens: トリム後の最大トークン数
        token_counter: トークン数をカウントするLLM

    Returns:
        結合・トリムされたメッセージリスト
    """
    # メッセージを結合
    combined_messages = add_messages(left_messages, right_messages)

    # メッセージをトリム
    trimmed_messages = trim_messages(
        combined_messages,
        max_tokens=MAX_TOKENS,
        token_counter=count_tokens_approximately,
        strategy="last",  # 最新のメッセージを保持
        include_system=True,  # システムメッセージを保持
    )

    return trimmed_messages

class CustomAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_and_trim_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
