from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# サブグラフのステート定義
class SubgraphState(TypedDict):
    messages: list[str]

def subgraph_node_1(state: SubgraphState):
    return {"messages": state.get("messages", []) + ["subgraph_node_1"]}

def subgraph_node_2(state: SubgraphState):
    return {"messages": state.get("messages", []) + ["subgraph_node_2"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("sub_node_1", subgraph_node_1)
subgraph_builder.add_node("sub_node_2", subgraph_node_2)
subgraph_builder.add_edge(START, "sub_node_1")
subgraph_builder.add_edge("sub_node_1", "sub_node_2")
subgraph_builder.add_edge("sub_node_2", END)
compiled_subgraph = subgraph_builder.compile()

# 親グラフのステート定義
class ParentState(TypedDict):
    messages: list[str]

def parent_node_1(state: ParentState):
    return {"messages": state.get("messages", []) + ["parent_node_1"]}

def parent_node_2(state: ParentState):
    return {"messages": state.get("messages", []) + ["parent_node_2"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("parent_node_1", parent_node_1)
parent_builder.add_node("parent_node_2", parent_node_2)

# サブグラフをノードとして追加
parent_builder.add_node("subgraph_node", compiled_subgraph)

parent_builder.add_edge(START, "parent_node_1")
parent_builder.add_edge("parent_node_1", "subgraph_node")
parent_builder.add_edge("subgraph_node", "parent_node_2")
parent_builder.add_edge("parent_node_2", END)
parent_graph = parent_builder.compile()

if __name__ == "__main__":
    # subgraphs=Trueを指定することで、サブグラフのノードも逐次実行される
    for chunk in parent_graph.stream({"messages": []}, stream_mode="values", subgraphs=True):
        print(chunk)
