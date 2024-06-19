from datetime import datetime
import os
import argparse
from functools import lru_cache
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from tavily import TavilyClient
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .envから環境変数を読み込む
    model_config = SettingsConfigDict(env_file=".env")

    # OpenAI APIキー
    OPENAI_API_KEY: str
    # Tavily APIキー
    TAVILY_API_KEY: str

    # gpt-4oを使用する場合の定数
    LLM_MODEL_NAME: str = "gpt-4o-2024-05-13"
    # gpt-3.5-turboを使用する場合の定数
    FAST_LLM_MODEL_NAME: str = "gpt-3.5-turbo-0125"
    # Tavilyの検索結果の最大数
    TAVILY_MAX_RESULTS: int = 5


settings = Settings()


# プロンプトファイルを読み込む関数
def load_prompt(name: str) -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.prompt")
    with open(prompt_path, "r") as f:
        return f.read()


# エージェントを実行する関数
def run_streaming_agent(agent, inputs):
    result = ""
    for s in agent.stream(inputs, stream_mode="values"):
        message: AIMessage = s["messages"][-1]
        if isinstance(message, tuple):
            result = str(message)
        else:
            result = message.content
            message.pretty_print()
    print(
        "================================= Final Result ================================="
    )
    print(result)
    return result


# 検索結果をサマリするためのチェイン
def summarize_search_chain() -> Runnable:
    llm = ChatOpenAI(model=settings.FAST_LLM_MODEL_NAME, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("user", "{query}")]
    ).partial(
        system=load_prompt("summarize_search_system"),
    )
    return prompt | llm | StrOutputParser()


# Tavilyクライアントのインスタンスを取得する関数
@lru_cache
def tavily_client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# 検索を実行するツール
@tool
def search(query: str) -> str:
    """Search for the given query and return the results as a string."""
    response = tavily_client().search(
        query, max_results=settings.TAVILY_MAX_RESULTS, include_raw_content=True
    )
    queries = []
    for document in response["results"]:
        # 1万字以上のコンテンツは省略
        # TavilyからPDFのパース結果等で大量の文字列が返ることがあるため
        if len(document["raw_content"]) > 10000:
            document["raw_content"] = document["raw_content"][:10000] + "..."
        queries.append(
            {
                "query": f"<source>\ntitle: {document['title']}\nurl: {document['url']}\ncontent: {document['raw_content']}\n</source>"
            }
        )
    # 検索結果をそれぞれ要約する
    summarize_results = summarize_search_chain().batch(queries)
    # 各要約結果にsourceタグを追加
    xml_results = []
    for result in summarize_results:
        xml_result = "<source>{}</source>".format(result)
        xml_results.append(xml_result)
    return "\n\n".join(xml_results)


# 成果物の内容がユーザー要求に対して十分かどうかをチェックするツール
@tool
def sufficiency_check(user_requirement: str, result: str) -> str:
    """Determine whether the answers generated adequately answer the question."""
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    user_prompt = f"ユーザーからの要求: {user_requirement}\n生成結果: {result}\n十分かどうかを判断してください。"
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("user", "{user}")]
    ).partial(
        system=load_prompt("sufficiency_classifier_system"),
        user=user_prompt,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


# レポートを生成するツール
@tool
def report_writer(user_requirement: str, source: str) -> str:
    """Generate reports based on user requests and sources of information gathered through searches."""
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    user_prompt = f"ユーザーからの要求: {user_requirement}\n情報源: {source}\n必ず情報源を基にユーザーからの要求を満たすレポートを生成してください。"
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("user", "{user}")]
    ).partial(
        system=load_prompt("report_writer_system"),
        user=user_prompt,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


# エージェントを作成する関数
def multi_step_agent():
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    # ツールとして検索、十分性チェック、レポート生成を指定
    tools = [search, sufficiency_check, report_writer]
    return create_react_agent(
        llm,
        tools=tools,
        messages_modifier=load_prompt("multi_step_answering_system").format(
            datetime.now().strftime("%Y年%m月%d日")
        ),
    )


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Process some queries.")
    parser.add_argument("--task", type=str, required=True, help="The query to search")

    # コマンドライン引数を解析
    args = parser.parse_args()

    # エージェントを実行
    inputs = {"messages": [("user", args.task)]}
    run_streaming_agent(multi_step_agent(), inputs)


if __name__ == "__main__":
    main()
