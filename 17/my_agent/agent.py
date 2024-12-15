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
from my_agent.task_executor_agent import \
    create_agent as create_task_executor_agent
from my_agent.task_planner_agent import \
    create_agent as create_task_planner_agent


class AgentInputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentPrivateState(TypedDict):
    tasks: list[str]
    results: list[str]


class AgentOutputState(TypedDict):
    final_output: str


class AgentState(AgentInputState, AgentPrivateState, AgentOutputState):
    pass


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
        self.reporter = Reporter(llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(
            state_schema=AgentState,
            input=AgentInputState,
            output=AgentOutputState,
        )

        graph.add_node("task_planner", create_task_planner_agent(self.llm))
        graph.add_node("task_executor", create_task_executor_agent(self.llm))
        graph.add_node("reporter", self.reporter)

        graph.add_edge(START, "task_planner")
        graph.add_edge("task_planner", "task_executor")
        graph.add_edge("task_executor", "reporter")
        graph.add_edge("reporter", END)

        return graph.compile(checkpointer=MemorySaver())


def create_agent(llm: ChatOpenAI) -> CompiledStateGraph:
    return Agent(llm).graph


graph = create_agent(ChatOpenAI(model="gpt-4o-mini"))

if __name__ == "__main__":
    png = graph.get_graph(xray=2).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png)
