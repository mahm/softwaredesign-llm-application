import operator
from datetime import datetime
from typing import Annotated, Any, Callable, Literal, Tuple

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.pregel.types import StateSnapshot
from pydantic import BaseModel, Field

APPROVE_TOKEN = "[APPROVE]"


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="分解されたタスクのリスト",
    )


class HumanInTheLoopAgentState(BaseModel):
    human_inputs: Annotated[list[str], operator.add] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    current_task_index: int = Field(default=0)
    results: list[str] = Field(default_factory=list)


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(
        self, human_inputs: list[str], latest_decomposed_tasks: list[str] | None = None
    ) -> DecomposedTasks:
        existing_tasks = latest_decomposed_tasks if latest_decomposed_tasks else []
        formatted_tasks = "\n".join([f"  - {task}" for task in existing_tasks])
        formatted_human_inputs = "\n".join(
            [f"  - {human_input}" for human_input in human_inputs]
        )
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "タスク: 与えられた目標またはユーザーフィードバックを反映し、具体的で実行可能なタスクに分解または改善してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "   - ユーザーのためのレポートを生成する。\n"
            "2. 作業内容は全てユーザーに共有されるため、ユーザーに情報を提出する必要はありません。\n"
            "3. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "4. タスクは実行可能な順序でリスト化すること。\n"
            "5. タスクは日本語で出力すること。\n"
            "6. 既存のタスクリストがある場合は、ユーザーフィードバックを最大限に反映させ、それを改善または補完してください。\n"
            "7. ユーザーフィードバックがある場合は、それを優先的に考慮し、タスクに反映させてください。\n"
            "8. タスクは必ず5個までにすること。\n"
            "既存のタスクリスト:\n"
            "{existing_tasks}\n\n"
            "目標またはユーザーフィードバック:\n"
            "{human_inputs}\n"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke(
            {"human_inputs": formatted_human_inputs, "existing_tasks": formatted_tasks}
        )


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: str, results: list[str]) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(self._create_task_message(task, results))
        return result["messages"][-1].content

    @staticmethod
    def _create_task_message(task: str, results: list[str]) -> dict[str, Any]:
        context = ""
        if results:
            context = "<context>\n"
            for i, result in enumerate(results, 1):
                context += f"<result_{i}>\n{result}\n</result_{i}>\n"
            context += "</context>\n\n"

        return {
            "messages": [
                (
                    "human",
                    f"{context}"
                    f"次のタスクを実行し、詳細な回答を提供してください。\n\nタスク: {task}\n\n"
                    "要件:\n"
                    "1. 必要に応じて提供されたツールを使用してください。\n"
                    "2. 実行は徹底的かつ包括的に行ってください。\n"
                    "3. 可能な限り具体的な事実やデータを提供してください。\n"
                    "4. 発見した内容を明確に要約してください。\n"
                    "5. <context>タグが存在する場合は、これまでの調査結果を参考にしてください。\n"
                    "6. 新しい情報を追加し、既存の情報を補完または更新してください。\n",
                )
            ]
        }


class HumanInTheLoopAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.subscribers: list[Callable[[str, str, str], None]] = []
        self.query_decomposer = QueryDecomposer(llm)
        self.task_executor = TaskExecutor(llm)
        self.graph = self._create_graph()

    def subscribe(self, subscriber: Callable[[str, str, str], None]) -> None:
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

        graph.add_edge(START, "decompose_query")
        graph.add_edge("decompose_query", "human_approval")
        graph.add_conditional_edges("human_approval", self._route_after_human_approval)
        graph.add_conditional_edges("execute_task", self._route_after_task_execution)

        memory = MemorySaver()

        return graph.compile(
            checkpointer=memory,
            interrupt_before=["human_approval"],
        )

    def _notify(
        self, type: Literal["human", "agent"], title: str, message: str
    ) -> None:
        for subscriber in self.subscribers:
            subscriber(type, title, message)

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
    ) -> Literal["execute_task", END]:
        is_task_completed = state.current_task_index >= len(state.tasks)
        if is_task_completed:
            return END
        else:
            return "execute_task"

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _decompose_query(self, state: HumanInTheLoopAgentState) -> DecomposedTasks:
        print("decompose_query")
        human_inputs = self._latest_human_inputs(state.human_inputs)
        # 初回のタスク分解時には過去のタスク分解結果を参考にしない
        if len(human_inputs) > 1:
            latest_decomposed_tasks = state.tasks
        else:
            latest_decomposed_tasks = []

        decomposed_tasks = self.query_decomposer.run(
            human_inputs=human_inputs, latest_decomposed_tasks=latest_decomposed_tasks
        )
        return {
            "tasks": decomposed_tasks.tasks,
            "current_task_index": 0,
            "results": [],
        }

    def _human_approval(self, state: HumanInTheLoopAgentState) -> dict:
        print("human_approval")
        pass

    def _execute_task(self, state: HumanInTheLoopAgentState) -> dict:
        print("execute_task")
        result = self.task_executor.run(
            task=state.tasks[state.current_task_index], results=state.results
        )
        return {
            "results": state.results + [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _stream_events(self, human_message: str | None, thread_id: str):
        if human_message:
            self._notify("human", human_message, "")
        for event in self.graph.stream(
            input=None,
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            message = ""
            # 実行ノードの情報を取得
            node = list(event.keys())[0]
            if node == "decompose_query":
                title, message = self._decompose_query_message(event[node])
            elif node == "execute_task":
                title, message = self._execute_task_message(event[node], thread_id)
            self._notify("agent", title, message)

    def _decompose_query_message(self, update_state: dict) -> Tuple[str, str]:
        tasks = "\n".join([f"- {task}" for task in update_state["tasks"]])
        return ("タスクを分解しました。", tasks)

    def _execute_task_message(
        self, update_state: dict, thread_id: str
    ) -> Tuple[str, str]:
        current_state = self._get_state(thread_id)
        current_task_index = update_state["current_task_index"] - 1
        executed_task = current_state.values["tasks"][current_task_index]
        result = update_state["results"][-1]
        return (executed_task, result)

    def _latest_human_inputs(self, human_inputs: list[str]) -> list[str]:
        # APPROVE_TOKENがある場合は、APPROVAL_TOKEN以降のリストを取得
        # それ以外は、リスト全体を取得
        if APPROVE_TOKEN in human_inputs:
            return human_inputs[human_inputs.index(APPROVE_TOKEN) + 1 :]
        else:
            return human_inputs
