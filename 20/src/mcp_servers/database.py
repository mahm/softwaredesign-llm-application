import datetime
import os
import sqlite3
from typing import Any, Dict

# SQLiteデータベースの永続化設定（環境変数DB_PATHが指定されていなければ "data.db" を使用）
DB_PATH = os.getenv("DB_PATH", "data.db")


def get_connection():
    """データベース接続を取得します"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """データベースの初期化と必要なテーブル・インデックスの作成を行います"""
    with get_connection() as conn:
        # search_resultsテーブルの作成
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,                     -- 検索クエリ
            source_url TEXT,                         -- 情報ソースURL
            title TEXT,                              -- コンテンツタイトル
            content TEXT,                            -- 抽出したコンテンツ
            summary TEXT,                            -- LLMが生成した要約
            content_type TEXT,                       -- 情報タイプ (ニュース/技術文書など)
            tags TEXT,                               -- タグ (カンマ区切り)
            reliability_score FLOAT,                 -- 信頼性スコア (0-1)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        )

        # インデックスの作成
        conn.execute(
            "CREATE INDEX IF NOT EXISTS index_search_results_on_query ON search_results(query);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS index_search_results_on_source_url ON search_results(source_url);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS index_search_results_on_content_type ON search_results(content_type);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS index_search_results_on_created_at ON search_results(created_at);"
        )


def save_search_result(
    query: str,
    url: str,
    title: str,
    content: str,
    content_type: str = "",
    summary: str = "",
    tags: str = "",
    reliability_score: float = 0.5,
) -> Dict[str, Any]:
    """
    検索結果をデータベースに保存します。

    引数:
        query: 検索クエリ
        url: 情報ソースのURL
        title: コンテンツのタイトル
        content: 抽出したコンテンツ（本文）
        content_type: 情報タイプ（例: "ニュース", "技術文書"）
        summary: 要約文（エージェントが生成）
        tags: カンマ区切りのタグ
        reliability_score: 信頼性スコア (0.0-1.0)

    返値:
        {"success": bool, "message": str, "result_id": Optional[int]}
    """
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with get_connection() as conn:
            # URLが既に保存されているか確認
            cur = conn.cursor()
            cur.execute("SELECT id FROM search_results WHERE source_url = ?", (url,))
            existing = cur.fetchone()

            if existing:
                # 既存エントリを更新
                conn.execute(
                    """
                UPDATE search_results 
                SET query = ?, title = ?, content = ?, content_type = ?, summary = ?, tags = ?, reliability_score = ?
                WHERE source_url = ?
                """,
                    (
                        query,
                        title,
                        content,
                        content_type,
                        summary,
                        tags,
                        reliability_score,
                        url,
                    ),
                )
                result_id = existing["id"]
                message = f"検索結果の更新に成功しました (ID: {result_id})"
            else:
                # 新規エントリを作成
                cur = conn.execute(
                    """
                INSERT INTO search_results 
                (query, source_url, title, content, summary, content_type, tags, reliability_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        query,
                        url,
                        title,
                        content,
                        summary,
                        content_type,
                        tags,
                        reliability_score,
                        now,
                    ),
                )
                result_id = cur.lastrowid
                message = f"検索結果の保存に成功しました (ID: {result_id})"

            return {"success": True, "message": message, "result_id": result_id}

    except Exception as e:
        return {"success": False, "message": f"保存エラー: {e}", "result_id": None}


def get_recent_results(
    days: int = 7, limit: int = 10, content_type: str = ""
) -> Dict[str, Any]:
    """
    指定された日数以内の最近の検索結果を取得します。

    引数:
        days: 何日前までの検索結果を取得するか (デフォルト: 7)
        limit: 返す結果の最大数 (デフォルト: 10)
        content_type: 特定のコンテンツタイプでフィルタリング (オプション)

    返値:
        {"success": bool, "message": str, "results": List[Dict]}
    """
    try:
        # days日前の日付を計算
        date_threshold = (
            datetime.datetime.now() - datetime.timedelta(days=days)
        ).strftime("%Y-%m-%d %H:%M:%S")

        # SQLクエリ構築
        sql = """
        SELECT id, query, source_url, title, summary, content_type, tags, reliability_score, created_at
        FROM search_results 
        WHERE created_at >= ?
        """
        params = [date_threshold]

        # content_typeが指定されていれば条件追加
        if content_type:
            sql += " AND content_type = ?"
            params.append(content_type)

        # 信頼性スコアの高い順にソート
        sql += " ORDER BY reliability_score DESC, created_at DESC LIMIT ?"
        params.append(str(limit))  # intをstrに変換

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

            if not rows:
                return {
                    "success": True,
                    "message": "指定された期間内の検索結果は見つかりませんでした。",
                    "results": [],
                }

            results = [dict(row) for row in rows]
            return {
                "success": True,
                "message": f"{len(results)}件の検索結果が見つかりました。",
                "results": results,
            }

    except Exception as e:
        return {"success": False, "message": f"検索結果取得エラー: {e}", "results": []}


def get_content_by_id(result_id: int) -> Dict[str, Any]:
    """
    特定IDの検索結果の詳細コンテンツを取得します。

    引数:
        result_id: 検索結果のID

    返値:
        {"success": bool, "message": str, "result": Optional[Dict]}
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
            SELECT id, query, source_url, title, content, summary, content_type, tags, 
                reliability_score, created_at
            FROM search_results 
            WHERE id = ?
            """,
                (result_id,),
            )

            result = cur.fetchone()
            if not result:
                return {
                    "success": False,
                    "message": f"エラー: ID {result_id} の検索結果が見つかりません。",
                    "result": None,
                }

            result_dict = dict(result)
            return {
                "success": True,
                "message": f"ID {result_id} の検索結果を取得しました。",
                "result": result_dict,
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"コンテンツ取得エラー: {e}",
            "result": None,
        }


def get_content_types() -> Dict[str, Any]:
    """
    データベースに保存されている全てのコンテンツタイプの一覧と各タイプの件数を返します。

    返値:
        {"success": bool, "message": str, "results": List[Dict]}
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
            SELECT content_type, COUNT(*) as count
            FROM search_results
            GROUP BY content_type
            ORDER BY count DESC
            """
            )

            rows = cur.fetchall()
            if not rows:
                return {
                    "success": True,
                    "message": "保存されている検索結果がありません。",
                    "results": [],
                }

            results = [dict(row) for row in rows]
            return {
                "success": True,
                "message": f"{len(results)}種類のコンテンツタイプが見つかりました。",
                "results": results,
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"コンテンツタイプ取得エラー: {e}",
            "results": [],
        }


def get_schema() -> Dict[str, Any]:
    """
    SQLiteデータベースのスキーマ情報（テーブル名と各カラム）を返します。

    返値:
        {"success": bool, "message": str, "schema": str}
    """
    try:
        schema_info = ""
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cur.fetchall()]

            for table in tables:
                schema_info += f"Table: {table}\n"
                cur.execute(f"PRAGMA table_info({table});")
                columns = cur.fetchall()
                for col in columns:
                    # col: (cid, name, type, notnull, dflt_value, pk)
                    schema_info += f"  - {col[1]} ({col[2]})\n"

        return {
            "success": True,
            "message": "スキーマ情報を取得しました。",
            "schema": schema_info,
        }

    except Exception as e:
        return {"success": False, "message": f"スキーマ取得エラー: {e}", "schema": ""}


def execute_select_query(query: str) -> Dict[str, Any]:
    """
    SQLiteデータベースに対してSELECTクエリを実行し、結果を返します。

    引数:
        query: 実行するSELECTクエリ

    返値:
        {"success": bool, "message": str, "results": List[Dict]}
    """
    if not query.strip().lower().startswith("select"):
        return {
            "success": False,
            "message": "Error: SELECT文のみ許可されています。",
            "results": [],
        }

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()

            results = [dict(row) for row in rows]
            return {
                "success": True,
                "message": f"{len(results)}件の結果が見つかりました。",
                "results": results,
            }

    except Exception as e:
        return {"success": False, "message": f"SELECTエラー: {e}", "results": []}


# 初期化処理
init_database()
