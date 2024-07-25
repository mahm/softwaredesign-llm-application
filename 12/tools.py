import os
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from settings import Settings
from tavily import TavilyClient
from utility import load_prompt

TAVILY_MAX_RESULTS = 5

settings = Settings()


# Tavilyクライアントのインスタンスを取得する関数
@lru_cache
def tavily_client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def summarize_search_chain() -> Runnable:
    llm = ChatOpenAI(model=settings.FAST_LLM_MODEL_NAME, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("user", "{query}")]
    ).partial(
        system=load_prompt("summarize_search_system"),
    )
    return prompt | llm | StrOutputParser()


# 検索を実行するツール
@tool
def search(query: str) -> str:
    """Search for the given query and return the results as a string."""
    response = tavily_client().search(
        query, max_results=TAVILY_MAX_RESULTS, include_raw_content=True
    )
    queries = []
    for document in response["results"]:
        # 1万字以上のコンテンツは省略
        if len(document["raw_content"]) > 10000:
            document["raw_content"] = document["raw_content"][:10000] + "..."
        queries.append(
            {
                "query": f"<source>\ntitle: {document['title']}\nurl: {document['url']}\ncontent: {document['raw_content']}\n</source>"
            }
        )
    summarize_results = summarize_search_chain().batch(queries)
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
