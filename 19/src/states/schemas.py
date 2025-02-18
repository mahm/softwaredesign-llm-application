from typing import Annotated, List, TypedDict

from typing_extensions import NotRequired


class SearchQuery(TypedDict):
    """検索クエリを表現するモデル"""

    query: str  # 自然な日本語の質問文
    reason: str  # この情報が必要な理由


class WriterResult(TypedDict):
    """記事生成の結果を表現するモデル"""

    content: str  # 生成された記事
    sources: List[str]  # 参照した情報源
    structure: List[str]  # 記事の構造（見出しなど）


class EvaluationResult(TypedDict):
    """評価結果を表現するモデル"""

    score: float  # 記事の評価スコア（0-100）
    improvement_points: List[str]  # 改善が必要な点のリスト
    required_searches: List[SearchQuery]  # 追加で必要な検索クエリのリスト


class RefinerState(TypedDict):
    """改善処理の入力となる状態"""

    content: str  # 改善対象の記事内容
    style: str  # 改善のスタイル


class RefinerResult(TypedDict):
    """改善結果を表現するモデル"""

    diff: str  # 改善内容のdiff
    style: str  # 適用された改善スタイル


class Revision(TypedDict):
    """記事の改訂を表現するモデル"""

    content: str  # 記事内容
    score: float  # 評価スコア（0-100）
    improvement_points: List[str]  # 改善ポイント
    timestamp: str  # 改善時刻（ISO形式）
    diffs: List[str]  # 生成されたdiffのリスト
    style: str  # 改善のスタイル（"保守的"、"中間"、"積極的"）


class CandidateResult(TypedDict):
    """改善候補の評価結果を表現するモデル"""

    content: str  # 候補の内容
    score: float  # 評価スコア
    improvement_points: List[str]  # 改善ポイント
    style: str  # 改善のスタイル
    iteration: int  # 生成されたイテレーション番号


def messages_reducer(current_messages: List[str], new_message: str) -> List[str]:
    """メッセージリストのReducer"""
    return current_messages + [new_message]


def candidates_reducer(
    current_candidates: List[CandidateResult], new_candidate: CandidateResult
) -> List[CandidateResult]:
    """候補リストのReducer"""
    return current_candidates + [new_candidate]


def revisions_reducer(
    current_revisions: List[Revision], new_revision: Revision
) -> List[Revision]:
    """改訂履歴のReducer"""
    return current_revisions + [new_revision]


class WebExplorerInputState(TypedDict):
    """入力状態を定義するクラス"""

    query: str  # ユーザーからの質問


class WebExplorerPrivateState(TypedDict):
    """非公開状態を定義するクラス"""

    candidates: Annotated[
        List[CandidateResult], candidates_reducer
    ]  # 評価済み改善候補リスト
    history: Annotated[List[Revision], revisions_reducer]  # 改訂履歴


class WebExplorerOutputState(TypedDict):
    """出力状態を定義するクラス"""

    final_article: NotRequired[str]  # 最終的な記事


class WebExplorerState(
    WebExplorerInputState, WebExplorerPrivateState, WebExplorerOutputState
):
    """Web探索エージェントの状態を定義するクラス"""

    pass
