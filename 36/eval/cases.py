"""ランナー成果物(results/)とar5iv本文からDeepEvalのテストケースを組み立てる。"""

import html as html_lib
import json
import re
import urllib.request
from pathlib import Path

from deepeval.test_case import LLMTestCase, ToolCall

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EXPECTED_TOOLS = [ToolCall(name="execute"), ToolCall(name="generate_pptx")]
MAX_SOURCE_CHARS = 120_000


def fetch_paper_text(arxiv_id: str) -> str:
    """ar5ivから論文本文をテキスト化して取得する"""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sd36-eval)"})
    html = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "ignore")
    # 数式はalttext(LaTeX)に置き換えて残す
    html = re.sub(
        r'(?is)<math[^>]*?alttext="([^"]*)"[^>]*>.*?</math>',
        lambda m: f" {html_lib.unescape(m.group(1))} ",
        html,
    )
    html = re.sub(r"(?is)<(script|style|math|nav|header|footer).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 参考文献以降は本文後半に出現した場合のみ打ち切る
    refs = list(re.finditer(r"\bReferences\b", text))
    if refs and refs[-1].start() > len(text) * 0.5:
        text = text[: refs[-1].start()]
    return text[:MAX_SOURCE_CHARS]


def slides_to_text(data: dict) -> str:
    """スライドJSONを評価用のプレーンテキストへ変換する。

    タイトルスライドは著者・所属・発表年などの書誌メタデータであり、
    要約の主張として判定させないため評価対象から除外する。
    """
    parts = [f"# {data['title']}"]
    for i, slide in enumerate(data["slides"], 1):
        if slide["type"] == "title":
            continue
        parts.append(f"## Slide {i} [{slide['type']}] {slide.get('title', '')}")
        if slide.get("subtitle"):
            parts.append(slide["subtitle"])
        for bullet in slide.get("bullets") or []:
            parts.append(f"- {bullet}")
    return "\n".join(parts)


def load_test_case(result_file: Path) -> LLMTestCase:
    run = json.loads(result_file.read_text())
    return LLMTestCase(
        name=f"{run['variant']}-{run['arxivId']}",
        input=fetch_paper_text(run["arxivId"]),
        actual_output=slides_to_text(run["slides"]),
        tools_called=[ToolCall(name=call["name"]) for call in run["toolCalls"]],
        expected_tools=EXPECTED_TOOLS,
    )


def load_test_cases(variant: str, arxiv_id: str | None = None) -> list[LLMTestCase]:
    pattern = f"{arxiv_id}.json" if arxiv_id else "*.json"
    files = sorted((RESULTS_DIR / variant).glob(pattern))
    return [load_test_case(file) for file in files]
