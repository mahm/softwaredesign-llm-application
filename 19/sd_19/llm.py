"""
LLMクライアントを提供するモジュール
"""

from typing import Any, Dict, TypeVar, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

T = TypeVar("T")

DEFAULT_MODEL = "claude-3-5-sonnet-latest"


def get_llm(model: str = DEFAULT_MODEL) -> ChatAnthropic:
    """OpenAI経由でLLMクライアントを取得する

    Args:
        model: モデル名

    Returns:
        ChatOpenAI: LLMクライアント
    """
    return ChatAnthropic(model=model, temperature=0.0)


def get_structured_llm(
    output_type: type[T], model: str = DEFAULT_MODEL
) -> Runnable[Dict[str, Any], T]:
    """構造化出力を行うLLMクライアントを取得する

    Args:
        output_type: 出力の型
        model: モデル名

    Returns:
        Runnable: 構造化出力を行うLLMクライアント
    """
    llm = get_llm(model).with_structured_output(output_type)
    return cast(Runnable[Dict[str, Any], T], llm)
