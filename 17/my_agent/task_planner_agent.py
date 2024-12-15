from datetime import datetime
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


class TaskPlannerAgentInputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class TaskPlannerAgentPrivateState(TypedDict):
    is_approved: bool
    reason: str

class TaskPlannerAgentOutputState(TypedDict):
    tasks: list[str]

class TaskPlannerAgentState(
    TaskPlannerAgentInputState,
    TaskPlannerAgentPrivateState,
    TaskPlannerAgentOutputState
):
    pass

class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_length=3,
        max_length=5,
        description="分解されたタスクのリスト",
    )

class TaskPlannerDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def __call__(self, state: TaskPlannerAgentState) -> dict:
        decomposed_tasks = self.run(
            messages=state.get("messages", []),
            latest_decomposed_tasks=state.get("tasks", []),
            reason=state.get("reason", "")
        )
        # 新しいタスクリストを作ったらreasonはクリアする
        return {"tasks": decomposed_tasks.tasks, "reason": ""}

    def run(
        self,
        messages: list[BaseMessage],
        latest_decomposed_tasks: list[str] | None,
        reason: str
    ) -> DecomposedTasks:
        existing_tasks = latest_decomposed_tasks if latest_decomposed_tasks else []
        formatted_tasks = "\n".join([f"  - {task}" for task in existing_tasks])
        formatted_messages = "\n".join([f"  - {message.content}" for message in messages])

        prompt_text = (
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
        )
        # 改善が必要な場合のみ、その旨と理由、既存のタスクリストを追加
        if reason:
            prompt_text += (
                "これまでに作成した検索タスクリストに改善が必要と判断されました。\n"
                f"改善が必要な理由: {reason}\n"
                "\n"
                "あなたはこの理由を参考に、検索タスクリストを再度分解・改善してください。\n"
                "\n"
                "既存の検索タスクリスト:\n"
                f"{existing_tasks}\n"
            )
        else:
            prompt_text += (
                "ユーザーの目標を達成するために必要な検索タスクを分解してください。\n"
            )

        prompt_text += (
            "\n"
            "要件:\n"
            "1. 全てのタスクはインターネット検索による情報収集のみとすること。\n"
            "2. 各検索タスクは以下の形式で記述すること：\n"
            "   「〜について検索して情報を収集する」\n"
            "3. 各検索タスクは具体的なキーワードや調査項目を含み、何を調べるべきか明確であること。\n"
            "4. 検索タスクは論理的な順序で配置すること。\n"
            "5. 検索タスクは日本語で記述し、必ず5個までにすること。\n\n"
            "ユーザーの目標:\n"
            "{messages}\n\n"
            "これらを考慮し、最適な検索タスクリストを生成してください。"
        )

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"messages": formatted_messages, "existing_tasks": formatted_tasks})


class TaskPlannerApprovalDecision(BaseModel):
    is_approved: bool = Field(description="Trueなら改善不要、Falseなら再改善")
    reason: str = Field(default="", description="改善が必要な場合、その理由を説明する")

class TaskPlannerCheckApproval:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def __call__(self, state: TaskPlannerAgentState) -> dict:
        decision = self.run(state.get("messages", []), state.get("tasks", []))
        return {"is_approved": decision.is_approved, "reason": decision.reason}

    def run(self, messages: list[BaseMessage], tasks: list[str]) -> TaskPlannerApprovalDecision:
        formatted_tasks = "\n".join([f"  - {task}" for task in tasks])
        messages_str = "\n".join([f"  - {message.content}" for message in messages])
        prompt = ChatPromptTemplate.from_template(
            "以下はユーザーの目標と、それを達成するために提案された検索タスクリストです。\n"
            "検索タスクリストを評価し、以下の基準で判断してください：\n\n"
            "評価基準：\n"
            "1. 全ての検索タスクが情報収集に焦点を当てているか\n"
            "2. 検索内容が具体的で明確か\n"
            "3. ユーザーの目標達成に必要な情報を網羅しているか\n\n"
            "上記の基準を全て満たし、ユーザーの目標達成に十分な検索タスクが含まれている場合は"
            "is_approvedをTrueにしてください。\n"
            "改善が必要な場合はFalseにし、どの基準が満たされていないか、"
            "どのように改善すべきか具体的な理由をreasonフィールドに記載してください。\n\n"
            "### ユーザーの目標\n"
            "{messages}\n\n"
            "### タスクリスト\n"
            "{tasks}\n"
        )
        chain = prompt | self.llm.with_structured_output(TaskPlannerApprovalDecision)
        return chain.invoke({"messages": messages_str, "tasks": formatted_tasks})

class TaskPlannerAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.task_decomposer = TaskPlannerDecomposer(llm)
        self.approval_checker = TaskPlannerCheckApproval(llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(
            state_schema=TaskPlannerAgentState,
            input=TaskPlannerAgentInputState,
            output=TaskPlannerAgentOutputState,
        )

        graph.add_node("decompose_query", self.task_decomposer)
        graph.add_node("check_approval", self.approval_checker)

        graph.add_edge(START, "decompose_query")
        graph.add_edge("decompose_query", "check_approval")
        graph.add_conditional_edges(
            "check_approval",
            lambda state: state["is_approved"],
            {
                True: END,
                False: "decompose_query",
            },
        )

        return graph.compile()

def create_agent(llm: ChatOpenAI) -> CompiledStateGraph:
    return TaskPlannerAgent(llm).graph

graph = create_agent(ChatOpenAI(model="gpt-4o-mini"))