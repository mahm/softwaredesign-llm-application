import os
import argparse
from functools import lru_cache
from typing import TypedDict, Annotated, Sequence
from operator import add
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from dotenv import load_dotenv
from retry import retry

# 環境変数を読み込む
load_dotenv()

# 定数の定義
LLM_MODEL_NAME = "gpt-4-turbo-2024-04-09"
TAVILY_MAX_RESULTS = 5


# タスクを表すデータクラス
class Task(BaseModel):
    id: int
    action: str
    description: str
    related_ids: list[int] = []


# タスクのリストを表すデータクラス
class Tasks(BaseModel):
    tasks: list[Task]


# 検索結果のコンテンツを表すデータクラス
class SearchContent(BaseModel):
    documents: list[dict]

    def __str__(self):
        return "\n\n".join(
            f"\"\"\"\ntitle: {item['title']}\nurl: {item['url']}\ncontent: {item['raw_content']}\n\"\"\""
            for item in self.documents
        )


# 成果物を表すデータクラス
class Artifact(BaseModel):
    id: int
    task: Task
    content: str | SearchContent

    def __str__(self):
        if self.task.action == "search":
            return "\n\n".join(
                [
                    f"### search '{self.task.description}' result:",
                    str(self.content),
                ]
            )
        else:
            return self.content


# エージェントの状態を表す型定義
class AgentState(TypedDict):
    query: str
    tasks: list[Task]
    artifacts: Annotated[Sequence[Artifact], add]
    next_task: Task | None
    next_node: str
    completed_task_ids: Annotated[Sequence[int], add]


# Tavilyクライアントのインスタンスを取得する関数
@lru_cache
def tavily_client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# プロンプトファイルを読み込む関数
def load_prompt(name: str) -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.prompt")
    with open(prompt_path, "r") as f:
        return f.read()


# 次のタスクを見つける関数
def find_next_task(tasks: list[Task], completed_task_ids: list[int]) -> Task | None:
    for task in tasks:
        if task.id not in completed_task_ids and all(
                related_id in completed_task_ids for related_id in task.related_ids
        ):
            return task
    return None


# 関連する成果物を取得する関数
def fetch_artifact(artifacts: list[Artifact], related_ids: list[int]) -> list[Artifact]:
    return [artifact for artifact in artifacts if artifact.id in related_ids]


# タスクを計画する関数
def plan(query: str) -> list[Task]:
    def dict_to_task(data: dict) -> list[Task]:
        return [
            Task(
                id=item["id"],
                action=item["action"],
                description=item["description"],
                related_ids=item.get("related_ids", []),
            )
            for item in data["properties"]
        ]

    @retry(tries=3)
    def invoke_chain(query: str) -> list[Task]:
        llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0).bind(
            response_format={"type": "json_object"}
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system_message}"), ("user", "{message}")]
        ).partial(system_message=load_prompt("plan_system"))
        chain = prompt | llm | JsonOutputParser(pydantic_model=Tasks) | dict_to_task
        return chain.invoke({"message": query})

    return invoke_chain(query)


# 検索を実行する関数
def search(query: str) -> list[dict]:
    response = tavily_client().search(query, max_results=TAVILY_MAX_RESULTS, include_raw_content=True)
    return response["results"]


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


# 調査グラフを表すクラス
class ResearchGraph:
    def __init__(self):
        self._graph = StateGraph(AgentState)
        self._graph.add_node("plan", self._run_plan)
        self._graph.add_node("search", self._run_search)
        self._graph.add_node("write", self._run_write)
        self._graph.add_node("route", self._run_route)
        self._graph.set_entry_point("plan")
        self._graph.add_edge("plan", "route")
        self._graph.add_edge("search", "route")
        self._graph.add_edge("write", "route")
        self._graph.add_conditional_edges(
            "route",
            self._router,
            {"search": "search", "write": "write", "end": END},
        )
        self._agent = self._graph.compile()

    # 計画を実行するメソッド
    def _run_plan(self, state: AgentState) -> dict:
        tasks = plan(state["query"])
        return {"tasks": tasks}

    # 検索を実行するメソッド
    def _run_search(self, state: AgentState) -> dict:
        current_task = state["next_task"]
        query = current_task.description
        documents = search(query)
        new_artifact = Artifact(
            id=current_task.id,
            task=current_task,
            content=SearchContent(documents=documents),
        )
        return {
            "artifacts": [new_artifact],
            "completed_task_ids": [current_task.id],
        }

    # 書き込みを実行するメソッド
    def _run_write(self, state: AgentState) -> dict:
        current_task = state["next_task"]
        query = current_task.description
        related_artifacts = fetch_artifact(
            list(state["artifacts"]), current_task.related_ids
        )
        documents = "\n\n".join(str(artifact.content) for artifact in related_artifacts)
        report = write(query, documents)
        new_artifact = Artifact(
            id=current_task.id, task=current_task, content=report
        )
        return {
            "artifacts": [new_artifact],
            "completed_task_ids": [current_task.id],
        }

    # ルーティングを実行するメソッド
    def _run_route(self, state: AgentState) -> dict:
        tasks = state["tasks"]
        completed_task_ids = state["completed_task_ids"]
        current_task = find_next_task(tasks, completed_task_ids)
        if current_task:
            return {"next_task": current_task, "next_node": current_task.action}
        else:
            return {"next_task": None, "next_node": "end"}

    # ルーターを定義するメソッド
    def _router(self, state: AgentState) -> str:
        return state["next_node"]

    # エージェントのプロパティ
    @property
    def agent(self):
        return self._agent


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Process some queries.')
    parser.add_argument('--query', type=str, required=True, help='The query to search')

    # コマンドライン引数を解析
    args = parser.parse_args()

    graph = ResearchGraph()
    query = args.query
    initial_state = {
        "query": query,
        "tasks": [],
        "documents": [],
        "next_task": None,
        "next_node": "",
        "completed_task_ids": []
    }
    final_output = None

    print("processing...\n\n")

    for s in graph.agent.stream(
            input=initial_state,
            config={"recursion_limit": 1000}
    ):
        if "plan" in s:
            print("## generated plan ##")
            tasks = s["plan"]["tasks"]
            for task in tasks:
                print(f"[{task.id}] {task.action}: {task.description}, related_ids: {task.related_ids}")
            print("\n")
        elif "route" in s:
            next_task = s["route"]["next_task"]
            if next_task:
                print(f"* [{next_task.id}] {next_task.action}: '{next_task.description}' processing...")
        elif "write" in s:
            final_output = s["write"]["artifacts"][0]
    print("\n\n")
    print("## final output ##")
    print(final_output)


if __name__ == "__main__":
    main()
