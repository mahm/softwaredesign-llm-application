import os
import argparse
from typing import Literal, TypedDict, Annotated, Sequence
from operator import add
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from single_step_approach import single_step_agent
from multi_step_approach import multi_step_agent
from settings import Settings
from utility import load_prompt

settings = Settings()


# 成果物を表すデータクラス
class Artifact(BaseModel):
    action: str
    content: str


# エージェントの状態を表す型定義
class AgentState(TypedDict):
    task: str
    method: Literal["A", "B", "C"]
    artifacts: Annotated[Sequence[Artifact], add]


# ユーザー要求に応じてエージェントの呼び出し分類を行う関数
def method_classifier(query: str) -> str:
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("user", "{query}")]
    ).partial(
        system=load_prompt("method_classifier_system"),
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})


# 非検索型QAを行う関数
def non_retrieval_qa(query: str) -> str:
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("user", "{query}")]
    ).partial(
        system=load_prompt("non_retrieval_qa_system"),
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})


class ResearchGraph:
    def __init__(self):
        self._graph = StateGraph(AgentState)
        self._graph.add_node("method_classifier", self._run_method_classifier)
        self._graph.add_node("non_retrieval_qa", self._run_non_retrieval_qa)
        self._graph.add_node("single_step_approach", self._run_single_step_approach)
        self._graph.add_node("multi_step_approach", self._run_multi_step_approach)

        self._graph.set_entry_point("method_classifier")
        self._graph.add_conditional_edges(
            "method_classifier",
            (lambda state: state["method"]),
            {
                "A": "non_retrieval_qa",
                "B": "single_step_approach",
                "C": "multi_step_approach",
            },
        )
        self._graph.add_edge("non_retrieval_qa", END)
        self._graph.add_edge("single_step_approach", END)
        self._graph.add_edge("multi_step_approach", END)

        self._agent = self._graph.compile()

    def _run_method_classifier(self, state: AgentState) -> dict:
        query = state["task"]
        method = method_classifier(query)
        return {"method": method}

    def _run_non_retrieval_qa(self, state: AgentState) -> dict:
        query = state["task"]
        result = non_retrieval_qa(query)
        artifact = Artifact(action="non_retrieval_qa", content=result)
        return {"artifacts": [artifact]}

    def _run_single_step_approach(self, state: AgentState) -> dict:
        query = state["task"]
        inputs = {"messages": [("user", query)]}
        result = single_step_agent().invoke(inputs)
        content = str(result["messages"][-1])
        artifact = Artifact(action="single_step_approach", content=content)
        return {"artifacts": [artifact]}

    def _run_multi_step_approach(self, state: AgentState) -> dict:
        query = state["task"]
        inputs = {"messages": [("user", query)]}
        result = multi_step_agent().invoke(inputs)
        content = str(result["messages"][-1])
        artifact = Artifact(action="multi_step_approach", content=content)
        return {"artifacts": [artifact]}

    @property
    def agent(self):
        return self._agent


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Process some queries.")
    parser.add_argument("--task", type=str, required=True, help="The query to search")

    # コマンドライン引数を解析
    args = parser.parse_args()

    graph = ResearchGraph()
    task = args.task
    initial_state = {
        "task": task,
        "artifacts": [],
    }

    print("processing...\n\n")

    for s in graph.agent.stream(input=initial_state, config={"recursion_limit": 1000}):
        for key, value in s.items():
            print(f"Node '{key}':")
            print(value)


if __name__ == "__main__":
    main()
