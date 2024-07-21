from operator import add
from typing import Annotated, Any, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from multi_step_approach import create_multi_step_agent
from single_step_approach import create_single_step_agent
from utility import load_prompt, run_invoke_agent


class Artifact(BaseModel):
    action: str = Field(..., description="実行されたアクション")
    content: str = Field(..., description="アクションの結果")


class AgentState(BaseModel):
    task: str = Field(..., description="ユーザーが入力したタスク")
    method: Literal["A", "B", "C"] = Field(
        default="A", description="選択された方法"
    )
    artifacts: Annotated[list[Artifact], add] = Field(
        default_factory=list, description="生成された成果物のリスト"
    )


class AdaptiveRagAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """グラフを作成し、ノードとエッジを設定する"""
        graph = StateGraph(AgentState)

        # ノードの追加
        graph.add_node("method_classifier", self._run_method_classifier)
        graph.add_node("non_retrieval_qa", self._run_non_retrieval_qa)
        graph.add_node("single_step_approach", self._run_single_step_approach)
        graph.add_node("multi_step_approach", self._run_multi_step_approach)

        # エッジの設定
        graph.set_entry_point("method_classifier")
        graph.add_conditional_edges(
            "method_classifier",
            lambda state: state.method,
            {
                "A": "non_retrieval_qa",
                "B": "single_step_approach",
                "C": "multi_step_approach",
            },
        )
        graph.add_edge("non_retrieval_qa", END)
        graph.add_edge("single_step_approach", END)
        graph.add_edge("multi_step_approach", END)

        return graph.compile()

    def _run_method_classifier(self, state: AgentState) -> dict[str, Any]:
        """メソッド分類器を実行する"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("method_classifier_system")), ("user", "{query}")]
        )
        chain = prompt | self.llm | StrOutputParser()
        method = chain.invoke({"query": state.task})
        return {"method": method}

    def _run_non_retrieval_qa(self, state: AgentState) -> dict[str, Any]:
        """非検索型QAを実行する"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("non_retrieval_qa_system")), ("user", "{query}")]
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state.task})
        artifact = Artifact(action="non_retrieval_qa", content=result)
        return {"artifacts": [artifact]}

    def _run_single_step_approach(self, state: AgentState) -> dict[str, Any]:
        """単一ステップアプローチを実行する"""
        inputs = {"messages": [("user", state.task)]}
        agent = create_single_step_agent(llm=self.llm)
        content = run_invoke_agent(agent, inputs)
        artifact = Artifact(action="single_step_approach", content=content)
        return {"artifacts": [artifact]}

    def _run_multi_step_approach(self, state: AgentState) -> dict[str, Any]:
        """複数ステップアプローチを実行する"""
        inputs = {"messages": [("user", state.task)]}
        agent = create_multi_step_agent(llm=self.llm)
        content = run_invoke_agent(agent, inputs)
        artifact = Artifact(action="multi_step_approach", content=content)
        return {"artifacts": [artifact]}

    def stream(self, task: str) -> str:
        """リサーチタスクを実行し、最終的な状態を返す"""
        initial_state = AgentState(task=task)
        final_output = ""
        for s in self.graph.stream(initial_state, stream_mode="values"):
            print(s)
            if s["artifacts"]:
                latest_artifact = s["artifacts"][-1]
                final_output = latest_artifact.content

        return final_output


def main():
    import argparse

    from settings import Settings

    # 共通の設定情報を読み込み
    settings = Settings()

    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="リサーチタスクを実行します")
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
    research_graph = AdaptiveRagAgent(llm=llm)
    research_graph.stream(args.task)


if __name__ == "__main__":
    main()
