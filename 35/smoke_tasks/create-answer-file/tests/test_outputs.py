from pathlib import Path


def test_answer_file_exact_content():
    assert Path("/app/answer.txt").read_text(encoding="utf-8").strip() == "terminal-bench harness"
