from datetime import datetime
from typing import Dict, Literal

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from ..chains import evaluator, refiner, writer
from ..states.schemas import (
    CandidateResult,
    EvaluationResult,
    RefinerResult,
    RefinerState,
    Revision,
    WebExplorerState,
    WriterResult,
)
from ..tools.tavily import search
from ..utils import apply_diff

# 定数
MAX_ITERATIONS = 3  # 最大反復回数
SUCCESS_THRESHOLD = 80.0  # 満足とみなすスコア閾値


def search_info(state: WebExplorerState) -> Command[Literal["candidate_generation"]]:
    """
    Webから情報を収集する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    result_str: str = search(query=state["query"], max_results=5)
    writer_result: WriterResult = writer.run(
        query=state["query"], search_result=result_str
    )

    # 初期のRevisionを作成
    initial_revision = Revision(
        content=writer_result["content"],
        score=0.0,
        improvement_points=[],
        timestamp=datetime.now().isoformat(),
        diffs=[],
        style="中間",
    )

    return Command(goto="candidate_generation", update={"history": [initial_revision]})


def spawn_candidates(state: WebExplorerState) -> Command[Literal["refine"]]:
    """
    3種類のリファインメントタスクを生成する

    Args:
        state: 現在の状態
    Returns:
        Command: 並列実行するリファインメントタスク
    """
    styles = ["保守的", "中間", "積極的"]
    current_revision = state["history"][-1]
    sends = [
        Send("refine", RefinerState(content=current_revision["content"], style=style))
        for style in styles
    ]
    return Command(goto=sends)


def refine_candidate(state: RefinerState) -> Command[Literal["apply_diff"]]:
    """
    指定されたスタイルで記事を改善し、diff形式で出力する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    refiner_result: RefinerResult = refiner.run(
        content=state["content"], style=state["style"]
    )
    return Command(
        goto="apply_diff",
        update={"diff": refiner_result["diff"], "style": refiner_result["style"]},
    )


def apply_diff_candidate(state: WebExplorerState) -> Command[Literal["evaluation"]]:
    """
    diffを元の文書に適用する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    current_revision = state["history"][-1]
    diff = str(state.get("diff", ""))
    style = str(state.get("style", "中間"))
    refined = apply_diff(current_revision["content"], diff)

    # 現在のイテレーション番号を計算
    iteration = len(state["history"])

    candidate = CandidateResult(
        content=refined,
        score=0.0,
        improvement_points=[],
        style=style,
        iteration=iteration,
    )

    return Command(goto="evaluation", update={"candidates": [candidate]})


def spawn_evaluations(state: WebExplorerState) -> Command[Literal["evaluate"]]:
    """
    候補記事の評価タスクを生成する

    Args:
        state: 現在の状態
    Returns:
        Command: 並列実行する評価タスク
    """
    candidates = state.get("candidates", [])
    current_iteration = len(state["history"])
    # 現在のイテレーションの候補のみを評価
    current_candidates = [c for c in candidates if c["iteration"] == current_iteration]

    return Command(
        goto=["evaluate"] * len(current_candidates),
        update=[{"content": candidate["content"]} for candidate in current_candidates],
    )


def evaluate_candidate(state: Dict[str, str]) -> Command[Literal["selection"]]:
    """
    記事を評価してスコアを返す

    Args:
        state: 評価対象の記事内容
    Returns:
        Command: 次のノードと状態の更新
    """
    evaluation: EvaluationResult = evaluator.run(content=state["content"])

    return Command(goto="selection", update={"evaluation": evaluation})


def select_candidate(state: WebExplorerState) -> Command[Literal["loop_decision"]]:
    """
    最高スコアの候補を選択する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    current_iteration = len(state["history"])
    current_candidates = [
        c for c in state["candidates"] if c["iteration"] == current_iteration
    ]

    if not current_candidates:
        return Command(goto="loop_decision")

    # 最高スコアの候補を選択
    best_candidate = max(current_candidates, key=lambda c: c["score"])

    # 新しいRevisionを作成
    new_revision = Revision(
        content=best_candidate["content"],
        score=best_candidate["score"],
        improvement_points=best_candidate["improvement_points"],
        timestamp=datetime.now().isoformat(),
        diffs=[],  # TODO: diffの保存
        style=best_candidate["style"],
    )

    return Command(
        goto="loop_decision", update={"history": [new_revision], "candidates": []}
    )


def loop_decision(
    state: WebExplorerState,
) -> Command[Literal["candidate_generation", END]]:  # type: ignore
    """
    ループを継続するか終了するかを判断する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    current_revision = state["history"][-1]
    current_score = current_revision["score"]
    current_iter = len(state["history"])

    if current_score >= SUCCESS_THRESHOLD or current_iter >= MAX_ITERATIONS:
        return Command(goto=END, update={"final_article": current_revision["content"]})
    else:
        return Command(goto="candidate_generation")


def create_web_explorer_workflow() -> StateGraph:
    """Web探索エージェントのワークフローを構築する"""
    workflow = StateGraph(WebExplorerState)

    # ノードの追加
    workflow.add_node("search", search_info)
    workflow.add_node("candidate_generation", spawn_candidates)
    workflow.add_node("refine", refine_candidate)
    workflow.add_node("apply_diff", apply_diff_candidate)
    workflow.add_node("evaluation", spawn_evaluations)
    workflow.add_node("evaluate", evaluate_candidate)
    workflow.add_node("selection", select_candidate)
    workflow.add_node("loop_decision", loop_decision)

    # エントリーポイントの設定
    workflow.set_entry_point("search")

    return workflow


graph = create_web_explorer_workflow()
