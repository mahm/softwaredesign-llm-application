import os
import argparse
from functools import lru_cache
from typing import TypedDict, Annotated, Sequence
from operator import add
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# 定数の定義
LLM_MODEL_NAME = "gpt-4o-2024-05-13"
TAVILY_MAX_RESULTS = 5


# 検索結果のコンテンツを表すデータクラス
class SearchContent(BaseModel):
    documents: list[dict]

    def __str__(self):
        return "\n\n".join(
            f"\"\"\"\ntitle: {item['title']}\nurl: {item['url']}\ncontent: {item['raw_content']}\n\"\"\""
            for item in self.documents
        )


# 評価結果のコンテンツを表すデータクラス
class EvaluationContent(BaseModel):
    score: float
    judge: str

    def __str__(self):
        return self.judge


# 成果物を表すデータクラス
class Artifact(BaseModel):
    action: str
    content: str | SearchContent | EvaluationContent

    def __str__(self):
        if self.task.action == "search":
            return "\n\n".join(
                [
                    f"### search result:",
                    str(self.content),
                ]
            )
        else:
            return self.content


# エージェントの状態を表す型定義
class AgentState(TypedDict):
    task: str
    refined_query: str
    artifacts: Annotated[Sequence[Artifact], add]


# Tavilyクライアントのインスタンスを取得する関数
@lru_cache
def tavily_client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# プロンプトファイルを読み込む関数
def load_prompt(name: str) -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.prompt")
    with open(prompt_path, "r") as f:
        return f.read()


# 関連する成果物を取得する関数
def retrieve_last_artifact(artifacts: list[Artifact], action: str) -> Artifact | None:
    reversed_artifacts = reversed(artifacts)
    selected_artifacts = (artifact for artifact in reversed_artifacts if artifact.action == action)
    return next(selected_artifacts, None)


# 検索を実行する関数
def search(query: str) -> list[dict]:
    response = tavily_client().search(query, max_results=TAVILY_MAX_RESULTS, include_raw_content=True)
    return response["results"]


@lru_cache
def load_cross_encoder(
        model_name: str = "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1",
        default_activation_function=None
):
    _cross_encoder = CrossEncoder(
        model_name,
        default_activation_function=default_activation_function,
    )
    _cross_encoder.max_length = 512
    return _cross_encoder


# 検索クエリと検索結果の関連度スコアの平均スコアを算出する
def evaluate(query: str, tavily_result: list[dict]) -> float:
    # 検索結果を文字列に変換
    documents = [record["title"] + " " + record["content"] for record in tavily_result]
    # Cross Encoderでスコアを算出
    ranks = load_cross_encoder().rank(query, documents)
    # スコアの平均を計算
    scores = [rank['score'] for rank in ranks]
    average_score = sum(scores) / len(scores)
    return average_score


# 書き込みを実行する関数
def write(task: str, documents: str) -> str:
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.7)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system_message}"), ("user", load_prompt("write_user"))]
    ).partial(
        system_message=load_prompt("write_system")
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"task": task, "documents": documents})


def query_refine(task: str, query: str | None, previous_score: float | None) -> str:
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("user", load_prompt("query_refine_user"))]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"task": task, "refined_query": query, "previous_score": previous_score})


# リサーチエージェントの実装
class ResearchGraph:
    def __init__(self):
        self._graph = StateGraph(AgentState)
        self._graph.add_node("query_refine", self._run_query_refine)
        self._graph.add_node("search", self._run_search)
        self._graph.add_node("evaluate", self._run_evaluate)
        self._graph.add_node("write", self._run_write)
        self._graph.set_entry_point("query_refine")
        self._graph.add_edge("query_refine", "search")
        self._graph.add_edge("search", "evaluate")
        self._graph.add_conditional_edges(
            "evaluate",
            self._router,
            {"query_refine": "query_refine", "write": "write"},
        )
        self._graph.add_edge("write", END)

        self._agent = self._graph.compile()

    # 計画を実行するメソッド
    def _run_query_refine(self, state: AgentState) -> dict:
        task = state["task"]
        refined_query = state["refined_query"]
        # 直近の評価結果を取得
        artifacts = state["artifacts"]
        relative_artifact = retrieve_last_artifact(artifacts, "evaluate")
        previous_score = relative_artifact.content.score if relative_artifact else None
        refined_query = query_refine(task, refined_query, previous_score)
        return {"refined_query": refined_query}

    # 検索を実行するメソッド
    def _run_search(self, state: AgentState) -> dict:
        query = state["refined_query"]
        documents = search(query)
        new_artifact = Artifact(
            action="search",
            content=SearchContent(documents=documents),
        )
        return {
            "artifacts": [new_artifact],
        }

    # 評価を実行するメソッド
    def _run_evaluate(self, state: AgentState) -> dict:
        refined_query = state["refined_query"]
        # 直近の検索結果を取得
        artifacts = state["artifacts"]
        relative_artifact = retrieve_last_artifact(artifacts, "search")
        # 検索結果のコンテンツを取得
        documents = relative_artifact.content.documents
        # 検索結果のコンテンツを評価
        score = evaluate(query=refined_query, tavily_result=documents)
        judge = "CORRECT" if score > 0.6 else "INCORRECT"

        # 評価結果を成果物として追加
        new_artifact = Artifact(
            action="evaluate",
            content=EvaluationContent(score=score, judge=judge),
        )
        return {
            "artifacts": [new_artifact],
        }

    # 書き込みを実行するメソッド
    def _run_write(self, state: AgentState) -> dict:
        task = state["task"]
        artifacts = state["artifacts"]
        relative_artifact = retrieve_last_artifact(artifacts, "search")
        report = write(task, relative_artifact.content.documents)
        new_artifact = Artifact(
            action="write",
            content=report
        )
        return {
            "artifacts": [new_artifact]
        }

    # ルーティングを実行するメソッド
    def _router(self, state: AgentState) -> str:
        artifacts = state["artifacts"]
        relative_artifact = retrieve_last_artifact(artifacts, "evaluate")
        if relative_artifact.content.judge == "CORRECT":
            return "write"
        else:
            return "query_refine"

    # エージェントのプロパティ
    @property
    def agent(self):
        return self._agent


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Process some queries.')
    parser.add_argument('--task', type=str, required=True, help='The query to search')

    # コマンドライン引数を解析
    args = parser.parse_args()

    graph = ResearchGraph()
    task = args.task
    initial_state = {
        "task": task,
        "artifacts": [],
    }

    print("processing...\n\n")

    for s in graph.agent.stream(
            input=initial_state,
            config={"recursion_limit": 1000}
    ):
        for key, value in s.items():
            print(f"Node '{key}':")
            if key == "search" or key == "write":
                print("search done")
            else:
                print(value)
        print("\n---\n")
    print("## final output ##")
    print(value["artifacts"][-1].content)


if __name__ == "__main__":
    main()
