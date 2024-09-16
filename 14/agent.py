import operator
from datetime import datetime
from typing import Annotated, Callable, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.types import StateSnapshot
from pydantic import BaseModel, Field

APPROVE_TOKEN = "[APPROVE]"


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list, description="分解されたタスクのリスト"
    )


class HumanInTheLoopAgentState(BaseModel):
    human_inputs: Annotated[list[str], operator.add] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    current_task_index: int = Field(default=0)
    current_node: str = Field(default="")
    results: Annotated[list[str], operator.add] = Field(default_factory=list)
    final_output: str = Field(default="")


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "   - ユーザーのためのレポートを生成する。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. タスクは日本語で出力すること。\n"
            "目標: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})


class HumanInTheLoopAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.subscribers: list[Callable[[str], None]] = []
        self.graph = self._create_graph()

    def subscribe(self, subscriber: Callable[[str], None]) -> None:
        self.subscribers.append(subscriber)

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        print("handle_human_message")
        if self.is_next_human_approval_node(thread_id):
            self.graph.update_state(
                config=self._config(thread_id),
                values={"human_inputs": [human_message]},
                as_node="human_approval",
            )
        else:
            self.graph.update_state(
                config=self._config(thread_id),
                values={"human_inputs": [human_message]},
                as_node=START,
            )
        self._stream_events(human_message=human_message, thread_id=thread_id)

    # def handle_approve(self, thread_id: str) -> None:
    #     print("handle_approve")
    #     self._stream_events(
    #         HumanInTheLoopAgentState(human_inputs=[APPROVE_TOKEN]), thread_id
    #     )

    def is_next_human_approval_node(self, thread_id: str) -> bool:
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_approval"

    def mermaid_png(self) -> bytes:
        return self.graph.get_graph().draw_mermaid_png()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(HumanInTheLoopAgentState)

        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("human_approval", self._human_approval)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)

        graph.add_edge(START, "decompose_query")
        graph.add_edge("decompose_query", "human_approval")
        graph.add_conditional_edges("human_approval", self._route_after_human_approval)
        graph.add_conditional_edges("execute_task", self._route_after_task_execution)
        graph.add_edge("aggregate_results", END)

        memory = MemorySaver()

        return graph.compile(
            checkpointer=memory,
            interrupt_before=["human_approval"],
        )

    def _notify(self, type: Literal["human", "agent"], message: str) -> None:
        for subscriber in self.subscribers:
            subscriber(type, message)

    def _route_after_human_approval(
        self, state: HumanInTheLoopAgentState
    ) -> Literal["decompose_query", "execute_task"]:
        is_human_approved = (
            state.human_inputs and state.human_inputs[-1] == APPROVE_TOKEN
        )
        if is_human_approved:
            return "execute_task"
        else:
            return "decompose_query"

    def _route_after_task_execution(
        self, state: HumanInTheLoopAgentState
    ) -> Literal["execute_task", "aggregate_results"]:
        is_task_completed = state.current_task_index >= len(state.tasks)
        if is_task_completed:
            return "aggregate_results"
        else:
            return "execute_task"

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _decompose_query(self, state: HumanInTheLoopAgentState) -> DecomposedTasks:
        print("decompose_query")
        return {"tasks": ["1", "2", "3"], "current_task_index": 0, "results": []}

    def _human_approval(self, state: HumanInTheLoopAgentState) -> dict:
        print("human_approval")
        pass

    def _execute_task(self, state: HumanInTheLoopAgentState) -> dict:
        print("execute_task")
        result = "executed"
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(self, state: HumanInTheLoopAgentState) -> dict:
        print("aggregate_results")
        return {"final_output": "completed"}

    def _stream_events(self, human_message: str | None, thread_id: str):
        if human_message:
            self._notify("human", human_message)
        for event in self.graph.stream(
            input=None,
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            message = ""
            # 実行ノードの情報を取得
            node = list(event.keys())[0]
            if node == "decompose_query":
                message = self._decompose_query_message(event[node])
            elif node == "execute_task":
                message = self._execute_task_message(event[node], thread_id)
            elif node == "aggregate_results":
                message = self._aggregate_results_message(event[node])
            self._notify("agent", message)

    def _decompose_query_message(self, update_state: dict) -> str:
        return f"タスクを分解しました。\n" f"タスク: {update_state['tasks']}\n"

    def _execute_task_message(self, update_state: dict, thread_id: str) -> str:
        current_state = self._get_state(thread_id)
        current_task_index = update_state["current_task_index"] - 1
        executed_task = current_state.values["tasks"][current_task_index]
        return f"タスクを実行します。\n" f"タスク: {executed_task}\n"

    def _aggregate_results_message(self, update_state: dict) -> str:
        return f"{update_state['final_output']}\n"
