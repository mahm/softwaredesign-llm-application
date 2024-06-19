from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .envから環境変数を読み込む
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRAINING_API_KEY: str = "false"
    LANGCHAIN_ENDPOINT: str = ""
    LANGCHAIN_PROJECT: str = ""
    LANGCHAIN_API_KEY: str = ""

    LLM_MODEL_NAME: str = "gpt-4o-2024-05-13"
    FAST_LLM_MODEL_NAME: str = "gpt-3.5-turbo-0125"
    TAVILY_MAX_RESULTS: int = 5
