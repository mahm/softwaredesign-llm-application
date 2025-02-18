"""
LLMクライアントを提供するモジュール
"""

from typing import Any, Dict, TypeVar, cast

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

T = TypeVar("T")


def get_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    """OpenAI経由でLLMクライアントを取得する

    Args:
        model: モデル名

    Returns:
        ChatOpenAI: LLMクライアント
    """
    return ChatOpenAI(model=model, temperature=0.0)


def get_structured_llm(
    output_type: type[T], model: str = "gpt-4o-mini"
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
