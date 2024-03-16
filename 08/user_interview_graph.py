import operator
from typing import Annotated, Sequence, TypedDict

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    mission: str
    persona: str
    messages: Annotated[Sequence[BaseMessage], operator.add]


class UserInterviewGraph:
    def __init__(self):
        self._model = ChatOpenAI(model="gpt-4-0125-preview")

        # グラフを初期化（AgentStateの型に従って情報が保存される）
        workflow = StateGraph(AgentState)

        # グラフのノードを追加
        workflow.add_node("question", self.generate_question)
        workflow.add_node("interview", self.generate_interview)
        workflow.add_node("report", self.generate_report)

        # グラフのエントリーポイントを設定
        workflow.set_entry_point("question")

        # ノードを接続するエッジを設定
        workflow.add_edge("question", "interview")
        workflow.add_edge("report", END)

        # インタビュー後、更にインタビューを続けるか判定する「条件付きエッジ」を設定
        workflow.add_conditional_edges(
            "interview", self.should_continue, {"continue": "question", "end": "report"}
        )

        # グラフをチェインとしてコンパイル
        self._agent = workflow.compile()

    @property
    def agent(self):
        return self._agent

    def execute_model(self, state, system_message, user_message):
        messages = state["messages"]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
                ("user", user_message),
            ]
        )

        chain = prompt | self._model
        return chain.invoke({"messages": messages})

    def generate_question(self, state):
        mission = state["mission"]
        system_message = "あなたはユーザーヒアリングのスペシャリストです。"
        user_message = f"ミッション「{mission}」に基づき、これまでのヒアリング内容を踏まえ、ユーザーへの質問を1件だけ100字以内で提示してください。"
        return {"messages": [self.execute_model(state, system_message, user_message)]}

    def generate_interview(self, state):
        persona = state["persona"]
        system_message = f"あなたは「{persona}」としてユーザーからの質問に100字以内で答えてください。あなたは演技のプロフェッショナルです。"
        user_message = state["messages"][-1].content
        return {"messages": [self.execute_model(state, system_message, user_message)]}

    def generate_report(self, state):
        mission = state["mission"]
        persona = state["persona"]
        system_message = "あなたは超有名コンサルタント会社のアソシエイトです。"
        user_message = f"ここまでのヒアリング内容を踏まえ、ミッション[{mission}」に基づき、「{persona}」のユーザーのニーズやインサイトをレポートしなさい。"
        return {"messages": [self.execute_model(state, system_message, user_message)]}

    def should_continue(self, state):
        if len(state["messages"]) < 9:
            return "continue"
        else:
            return "end"


if __name__ == "__main__":
    graph = UserInterviewGraph()
    print("-- start user interview --")
    for s in graph.agent.stream(
        {
            "mission": "運動についての意識調査",
            "persona": "40代の会社員、男性、システムエンジニア、運動は滅多にしない",
            "messages": [],
        }
    ):
        if "question" in s:
            content = s["question"]["messages"][0].content
            print(f"質問: {content}", flush=True)
        if "interview" in s:
            content = s["interview"]["messages"][0].content
            print(f"答え: {content}\n", flush=True)
        if "report" in s:
            content = s["report"]["messages"][0].content
            print("-- report --")
            print(f"{content}", flush=True)
