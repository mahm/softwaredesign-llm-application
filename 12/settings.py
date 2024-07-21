from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .envから環境変数を読み込む
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = ""
    LANGCHAIN_API_KEY: str = ""

    LLM_MODEL_NAME: str = "gpt-4o-2024-05-13"
    FAST_LLM_MODEL_NAME: str = "gpt-4o-mini-2024-07-18"
    TAVILY_MAX_RESULTS: int = 5
