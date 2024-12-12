import operator
from datetime import datetime
from typing import Annotated, Callable, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from my_agent.task_agent import create_agent as create_task_agent


class AgentInputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentPrivateState(TypedDict):
    tasks: list[str]
    results: list[str]
    is_approved: bool


class AgentOutputState(TypedDict):
    final_output: str


class AgentState(AgentInputState, AgentPrivateState, AgentOutputState):
    pass


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="分解されたタスクのリスト",
    )


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def __call__(self, state: AgentState) -> dict:
        decomposed_tasks = self.run(state.get("messages", []), state.get("tasks", []))
        return {"tasks": decomposed_tasks.tasks}

    def run(
        self,
        messages: list[BaseMessage],
        latest_decomposed_tasks: list[str] | None = None,
    ) -> DecomposedTasks:
        existing_tasks = latest_decomposed_tasks if latest_decomposed_tasks else []
        formatted_tasks = "\n".join([f"  - {task}" for task in existing_tasks])
        formatted_messages = "\n".join(
            [f"  - {message.content}" for message in messages]
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
            "{messages}\n"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke(
            {"messages": formatted_messages, "existing_tasks": formatted_tasks}
        )


class ApprovalStatus(BaseModel):
    is_approved: bool = Field(
        description="タスク分解結果が承認されたかどうかを判断してください。"
    )


class CheckApproval:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        approval_status = self.run(state.get("messages", []))
        return {"is_approved": approval_status.is_approved}

    def run(self, messages: list[BaseMessage]) -> ApprovalStatus:
        prompt = ChatPromptTemplate.from_template(
            "次の会話履歴から判断して、エージェントによるタスク分解結果が承認されたかを判断してください。\n\n"
            "### 会話履歴\n"
            "{messages}\n"
        )
        chain = prompt | self.llm.with_structured_output(ApprovalStatus)
        messages_str = "\n".join([f"  - {message.content}" for message in messages])
        return chain.invoke({"messages": messages_str})


class Reporter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def __call__(self, state: AgentPrivateState) -> dict:
        return {"final_output": self.run(state.get("results", []))}

    def run(self, results: list[str]) -> str:
        prompt = ChatPromptTemplate.from_template(
            "### 調査結果\n"
            "{results}\n\n"
            "### タスク\n"
            "調査結果に基づき、調査結果の内容を漏れなく整理したレポートを作成してください。"
        )
        results_str = "\n\n".join(
            f"Info {i+1}:\n{result}" for i, result in enumerate(results)
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"results": results_str})


class Agent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.subscribers: list[Callable[[str, str, str], None]] = []
        self.query_decomposer = QueryDecomposer(llm)
        self.check_approval = CheckApproval(llm)
        self.reporter = Reporter(llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(
            state_schema=AgentState,
            input=AgentInputState,
            output=AgentOutputState,
        )

        graph.add_node("decompose_query", self.query_decomposer)
        graph.add_node("human_feedback", self._human_feedback)
        graph.add_node("check_approval", self.check_approval)
        graph.add_node("execute_task", create_task_agent(self.llm))
        graph.add_node("reporter", self.reporter)

        graph.add_edge(START, "decompose_query")
        graph.add_edge("decompose_query", "human_feedback")
        graph.add_edge("human_feedback", "check_approval")
        graph.add_conditional_edges(
            "check_approval",
            lambda state: state["is_approved"],
            {
                True: "execute_task",
                False: "decompose_query",
            },
        )
        graph.add_edge("execute_task", "reporter")
        graph.add_edge("reporter", END)

        memory = MemorySaver()

        return graph.compile(
            checkpointer=memory,
            interrupt_before=["human_feedback"],
        )

    def _human_feedback(self, state: AgentState) -> dict:
        pass


def create_agent(llm: ChatOpenAI) -> CompiledStateGraph:
    return Agent(llm).graph


graph = create_agent(ChatOpenAI(model="gpt-4o-mini"))
