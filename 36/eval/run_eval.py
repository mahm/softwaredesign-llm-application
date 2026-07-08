"""results/<variant>/ のランナー成果物をDeepEvalで評価する。

使い方:
    uv run eval/run_eval.py baseline
    uv run eval/run_eval.py improved
    uv run eval/run_eval.py baseline --id 1706.03762 --repeat 3  # judgeのブレ確認
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from cases import load_test_cases  # noqa: E402
from deepeval import evaluate  # noqa: E402
from metrics import build_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["baseline", "improved"])
    parser.add_argument("--id", help="特定のarXiv IDだけ評価する")
    parser.add_argument("--repeat", type=int, default=1, help="judgeのブレ確認用の繰り返し回数")
    args = parser.parse_args()

    test_cases = load_test_cases(args.variant, args.id)
    print(f"[info] variant={args.variant} cases={[c.name for c in test_cases]}")

    runs = []
    for _ in range(args.repeat):
        result = evaluate(test_cases=test_cases, metrics=build_metrics())
        runs.append(
            {
                test_result.name: {
                    metric.name: {"score": metric.score, "reason": metric.reason}
                    for metric in test_result.metrics_data
                }
                for test_result in result.test_results
            }
        )

    suffix = f"-{args.id}" if args.id else ""
    out_file = ROOT / "results" / "eval" / f"{args.variant}{suffix}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps({"variant": args.variant, "runs": runs}, ensure_ascii=False, indent=2) + "\n"
    )

    for i, run in enumerate(runs, 1):
        print(f"=== run {i} ===")
        for case_name, case_metrics in run.items():
            for metric_name, data in case_metrics.items():
                print(f"  {case_name}  {metric_name}: {data['score']:.3f}")
    print(f"[saved] {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
