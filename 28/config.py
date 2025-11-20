"""
環境変数設定ヘルパー
OpenAI/Azure OpenAIの切り替えとモデル設定を管理
"""

import os
from dotenv import load_dotenv
import dspy # type: ignore

load_dotenv()

# プロバイダー設定
PROVIDER_NAME = os.getenv("PROVIDER_NAME", "openai")

# LLMモデル設定
SMART_MODEL = os.getenv("SMART_MODEL", "gpt-4.1")
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4.1-nano")
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4.1-mini")  # LLM as a Judge用評価モデル

# 埋め込みモデル設定
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 検索設定
RETRIEVAL_K = 10  # 検索結果の取得数


def configure_lm(model_name: str | None = None, temperature: float = 0.0, max_tokens: int = 4096) -> dspy.LM:
    """DSPy用のLM設定を作成"""
    if model_name is None:
        model_name = FAST_MODEL

    if PROVIDER_NAME == "azure":
        # Azure OpenAIの設定
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        return dspy.LM(
            model=f"azure/{model_name}",
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        # OpenAIの設定
        api_key = os.getenv("OPENAI_API_KEY")

        return dspy.LM(
            model=f"openai/{model_name}",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )


def configure_embedder() -> dspy.Embedder:
    """DSPy用の埋め込みモデル設定を作成"""
    if PROVIDER_NAME == "azure":
        # Azure OpenAIの埋め込み設定
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        return dspy.Embedder(
            model=f"azure/{EMBEDDING_MODEL}",
            api_base=api_base,
            api_key=api_key
        )
    else:
        # OpenAIの埋め込み設定
        api_key = os.getenv("OPENAI_API_KEY")

        return dspy.Embedder(
            model=f"openai/{EMBEDDING_MODEL}",
            api_key=api_key
        )
