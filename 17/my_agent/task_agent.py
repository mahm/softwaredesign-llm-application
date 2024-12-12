import operator
from typing import Annotated, Sequence, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send


class TaskAgentInputState(TypedDict):
    tasks: list[str]


class TaskAgentOutputState(TypedDict):
    results: Annotated[Sequence[str], operator.add]


class TaskAgentState(TaskAgentInputState, TaskAgentOutputState):
    pass


# 並列ノードに値を受け渡すためのState
class ParallelState(TypedDict):
    task: str
    results: Annotated[Sequence[str], operator.add]


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

    def __call__(self, state: ParallelState) -> dict:
        return {"results": [self.run(state["task"])]}

    def run(self, task: str) -> str:
        messages = {
            "messages": [
                (
                    "human",
                    f"次のタスクを実行し、詳細な回答を提供してください。\n\nタスク: {task}\n\n"
                    "要件:\n"
                    "1. 必要に応じて提供されたツールを使用してください。\n"
                    "2. 実行は徹底的かつ包括的に行ってください。\n"
                    "3. 可能な限り具体的な事実やデータを提供してください。\n"
                    "4. 発見した内容を明確に要約してください。\n",
                )
            ]
        }
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(messages)
        return result["messages"][-1].content


class TaskAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.task_executor = TaskExecutor(self.llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(
            state_schema=TaskAgentState,
            input=TaskAgentInputState,
            output=TaskAgentOutputState,
        )
        graph.add_node("execute_task", self.task_executor)
        graph.add_conditional_edges(
            START, self.routing_parallel_nodes, ["execute_task"]
        )
        graph.add_edge("execute_task", END)

        return graph.compile()

    def routing_parallel_nodes(self, state: TaskAgentInputState) -> list[Send]:
        return [Send("execute_task", {"task": task}) for task in state.get("tasks", [])]


def create_agent(llm: ChatOpenAI) -> CompiledStateGraph:
    return TaskAgent(llm).graph


graph = create_agent(ChatOpenAI(model="gpt-4o-mini"))
