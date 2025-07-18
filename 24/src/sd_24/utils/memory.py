from typing import Any
from langchain_anthropic import ChatAnthropic


class Memory:
    """Claude 3.5 Haikuを使用したインテリジェント圧縮機能付き外部メモリ"""

    def __init__(self):
        self._store: dict[str, Any] = {}
        # 圧縮用のClaude 3.5 Haiku
        self._compressor = ChatAnthropic(
            model_name="claude-3-5-haiku-20241022", temperature=0
        )

    def set(self, key: str, value: Any) -> None:
        """データを保存（圧縮は明示的に指定）"""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """データを取得"""
        return self._store.get(key, default)

    async def compress_research(self, topic: str, findings: str) -> str:
        """Web検索結果をインテリジェントに圧縮"""
        prompt = f"""
        以下はWeb検索で得られた「{topic}」に関する情報です。
        重要なポイントを抓えて簡潔に要約してください：
        
        {findings}
        """
        response = await self._compressor.ainvoke(prompt)
        if isinstance(response.content, str):
            return response.content
        else:
            # リストの場合は結合
            return "\n".join(str(item) for item in response.content)


# シングルトンインスタンスとして公開
memory = Memory()
