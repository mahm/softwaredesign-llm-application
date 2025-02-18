import operator
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


class EvaluationResult(TypedDict, total=False):
    """評価結果を表現するモデル"""

    score: float  # 記事の評価スコア（0-100）
    improvement_points: List[str]  # 改善が必要な点のリスト
    required_searches: List[SearchQuery]  # 追加で必要な検索クエリのリスト
    selected_index: int  # 選択された改善案のインデックス（0-based）
    selection_reason: str  # 選択理由


class RefinerState(TypedDict):
    """改善処理の入力となる状態"""

    content: str  # 改善対象の記事内容
    style: str  # 改善のスタイル
    search_results: str  # 追加の検索結果
    iteration: int  # 現在のイテレーション番号


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
    """改善候補を表現するモデル"""

    style: str  # 改善のスタイル（"保守的"、"中間"、"積極的"）
    diff: str  # 改善内容のdiff
    iteration: int  # 生成されたイテレーション番号


class WebExplorerInputState(TypedDict):
    """入力状態を定義するクラス"""

    query: str  # ユーザーからの質問


class WebExplorerPrivateState(TypedDict):
    """非公開状態を定義するクラス"""

    draft: str  # 初稿
    candidates: Annotated[List[CandidateResult], operator.add]  # 評価済み改善候補リスト
    revisions: Annotated[List[Revision], operator.add]  # 改訂履歴


class WebExplorerOutputState(TypedDict):
    """出力状態を定義するクラス"""

    final_article: NotRequired[str]  # 最終的な記事


class WebExplorerState(
    WebExplorerInputState, WebExplorerPrivateState, WebExplorerOutputState
):
    """Web探索エージェントの状態を定義するクラス"""

    pass
