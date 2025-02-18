from datetime import datetime
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from ..chains import evaluator, refiner, writer
from ..states.schemas import (
    CandidateResult,
    EvaluationResult,
    RefinerState,
    Revision,
    WebExplorerInputState,
    WebExplorerOutputState,
    WebExplorerState,
)
from ..tools.tavily import search
from ..utils import apply_patch

# 定数
MAX_ITERATIONS = 3  # 最大反復回数
SUCCESS_THRESHOLD = 80.0  # 満足とみなすスコア閾値


def create_draft(state: WebExplorerState) -> Command[Literal["evaluate_draft"]]:
    """
    初稿を作成する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    # 検索を実行
    search_result = search(query=state["query"], max_results=5)

    # 記事を生成
    writer_result: str = writer.run(query=state["query"], search_result=search_result)

    return Command(
        goto="evaluate_draft",
        update={"draft": writer_result},
    )


def spawn_refinements(state: WebExplorerState) -> Command[Literal["create_diff"]]:
    """
    3種類の改善タスクを生成する

    Args:
        state: 現在の状態
    Returns:
        Command: 並列実行する改善タスク
    """
    styles = ["保守的", "中間", "積極的"]
    current_revision = state["revisions"][-1]
    current_iteration = len(state["revisions"])

    # 改善点に基づいて検索を実行
    search_results_text = ""
    if current_revision["improvement_points"]:
        search_results = [
            search(query=point, max_results=5)
            for point in current_revision["improvement_points"]
        ]
        search_results_text = "\n\n".join(search_results)

    # 各スタイルの改善タスクを生成
    sends = [
        Send(
            "create_diff",
            RefinerState(
                content=current_revision["content"],
                style=style,
                search_results=search_results_text,
                iteration=current_iteration,
            ),
        )
        for style in styles
    ]
    return Command(goto=sends)


def create_diff(state: RefinerState) -> Command[Literal["evaluate_refinement"]]:
    """
    指定されたスタイルで改善案をdiff形式で生成する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    style = state["style"]
    search_results = state["search_results"]
    iteration = state["iteration"]

    diff: str = refiner.run(
        draft=state["content"],
        style=style,
        search_results=search_results,
    )

    # 改善案を候補として保存
    candidate = CandidateResult(
        style=style,
        diff=diff,
        iteration=iteration,
    )

    return Command(
        goto="evaluate_refinement",
        update={
            "candidates": [candidate],  # operator.addによって自動的に結合される
        },
    )


def evaluate_draft(state: WebExplorerState) -> Command[Literal["check_completion"]]:
    """
    初稿を評価する

    Args:
        state: 評価対象の記事内容
    Returns:
        Command: 次のノードと状態の更新
    """
    draft = state["draft"]
    evaluation: EvaluationResult = evaluator.run(
        content=draft, threshold=SUCCESS_THRESHOLD
    )

    # 評価結果をRevisionに反映
    initial_revision = Revision(
        content=draft,
        score=evaluation["score"],
        improvement_points=evaluation["improvement_points"],
        timestamp=datetime.now().isoformat(),
        diffs=[],
        style="中間",
    )

    return Command(
        goto="check_completion",
        update={
            "revisions": [initial_revision],
        },
    )


def evaluate_refinement(
    state: WebExplorerState,
) -> Command[Literal["check_completion"]]:
    """
    改善案（diff）を評価する

    Args:
        state: 評価対象のdiff
    Returns:
        Command: 次のノードと状態の更新
    """
    current_revision = state["revisions"][-1]
    current_iteration = len(state["revisions"])

    # 現在のイテレーションの候補のみを評価
    current_candidates = [
        c for c in state["candidates"] if c["iteration"] == current_iteration
    ]

    # 現在の記事と改善案を一つの文章に結合
    combined_content = "\n\n".join(
        [f"現在の記事：\n{current_revision['content']}"]
        + [
            f"改善案{i+1}（{c['style']}スタイル）：\n{c['diff']}"
            for i, c in enumerate(current_candidates)
        ]
    )

    # 一括評価を実行
    evaluation: EvaluationResult = evaluator.run(content=combined_content)

    # 選択された改善案の情報を取得
    selected = current_candidates[evaluation["selected_index"]]

    # 選択された改善案を適用
    refined = apply_patch(current_revision["content"], str(selected["diff"]))

    # 新しいRevisionを作成
    new_revision = Revision(
        content=refined,
        score=evaluation["score"],
        improvement_points=evaluation["improvement_points"],
        timestamp=datetime.now().isoformat(),
        diffs=[str(selected["diff"])],
        style=str(selected["style"]),
    )

    return Command(
        goto="check_completion",
        update={
            "revisions": [new_revision],
        },
    )


def check_completion(
    state: WebExplorerState,
) -> Command[Literal["spawn_refinements", END]]:  # type: ignore
    """
    改善を継続するか完了するかを判断する

    Args:
        state: 現在の状態
    Returns:
        Command: 次のノードと状態の更新
    """
    current_revision = state["revisions"][-1]
    current_score = current_revision["score"]
    current_iter = len(state["revisions"])

    if current_score >= SUCCESS_THRESHOLD or current_iter >= MAX_ITERATIONS:
        return Command(goto=END, update={"final_article": current_revision["content"]})
    else:
        return Command(goto="spawn_refinements")


def create_web_explorer_workflow() -> StateGraph:
    """Web探索エージェントのワークフローを構築する"""
    workflow = StateGraph(
        config_schema=WebExplorerState,
        input=WebExplorerInputState,
        output=WebExplorerOutputState,
    )

    # ノードの追加
    workflow.add_node(create_draft)
    workflow.add_node(evaluate_draft)
    workflow.add_node(spawn_refinements)
    workflow.add_node(create_diff)
    workflow.add_node(evaluate_refinement)
    workflow.add_node(check_completion)

    # エントリーポイントの設定
    workflow.set_entry_point("create_draft")

    return workflow


graph = create_web_explorer_workflow()
