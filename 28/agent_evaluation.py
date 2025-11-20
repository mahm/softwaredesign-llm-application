# -*- coding: utf-8 -*-
"""
File Exploration Agent Evaluation Script

This script evaluates the file exploration agent on the test dataset using LLM as a Judge.
"""

import os
import sys
import argparse
import dspy
from datetime import datetime

from config import configure_lm, SMART_MODEL, FAST_MODEL, EVAL_MODEL
from agent_module import FileExplorationAgent
from dataset_loader import load_file_exploration_dataset
from agent_optimization_gepa import create_llm_judge_metric, ReportEvaluation


# Optimized model path (symlink to latest)
GEPA_OPTIMIZED_MODEL_LATEST = "artifact/agent_gepa_optimized_latest.json"


def evaluate_agent_detailed(agent, examples, metric, eval_lm, agent_name="Agent"):
    """
    Evaluate agent with detailed output.

    Args:
        agent: Agent to evaluate
        examples: List of evaluation examples
        metric: Evaluation metric function
        eval_lm: LM for evaluation (LLM as a Judge, e.g., gpt-4.1-mini)
        agent_name: Name of agent for display

    Returns:
        tuple: (average_score, detailed_results)
    """
    print(f"\n{'=' * 80}")
    print(f"{agent_name} Evaluation")
    print(f"{'=' * 80}")

    detailed_results = []
    scores = []

    for i, ex in enumerate(examples, 1):
        print(f"\n--- Test Example {i}/{len(examples)} ---")
        print(f"Task: {ex.task[:100]}...")
        print(f"Difficulty: {ex.difficulty}")

        # Run agent (uses globally configured LM)
        pred = agent(task=ex.task, working_directory=ex.working_directory)

        # Evaluate with LLM as a Judge (metric function handles LM internally)
        score = metric(ex, pred)
        scores.append(score)

        # Create evaluator for detailed feedback
        evaluator = dspy.ChainOfThought(ReportEvaluation)

        with dspy.context(lm=eval_lm):
            eval_result = evaluator(
                task=ex.task,
                report=pred.report if hasattr(pred, 'report') else "",
                criteria=ex.criteria
            )

        # Display results
        print(f"\nScore: {score:.2f} ({eval_result.score}/10)")
        print(f"Explanation: {eval_result.explanation[:200]}...")
        if hasattr(eval_result, 'improvement_suggestions') and eval_result.improvement_suggestions:
            print(f"Improvement: {eval_result.improvement_suggestions[:150]}...")

        detailed_results.append({
            "task": ex.task,
            "difficulty": ex.difficulty,
            "score": score,
            "raw_score": eval_result.score,
            "explanation": eval_result.explanation,
            "improvement_suggestions": eval_result.improvement_suggestions if hasattr(eval_result, 'improvement_suggestions') else "",
            "report": pred.report if hasattr(pred, 'report') else ""
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'=' * 80}")
    print(f"{agent_name} Average Score: {avg_score:.3f}")
    print(f"{'=' * 80}\n")

    return avg_score, detailed_results


def save_evaluation_report(baseline_results, optimized_results, baseline_avg, optimized_avg, timestamp):
    """
    Save detailed evaluation report to file.

    Args:
        baseline_results: Baseline evaluation results
        optimized_results: Optimized evaluation results
        baseline_avg: Baseline average score
        optimized_avg: Optimized average score
        timestamp: Timestamp string
    """
    os.makedirs("tmp/reports", exist_ok=True)
    report_path = os.path.join("tmp/reports", f"test_evaluation_{timestamp}.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# File Exploration Agent - Test Set Evaluation Report\n\n")
        f.write(f"**Evaluation Date**: {timestamp}\n")
        f.write(f"**Test Examples**: {len(baseline_results)}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Baseline Average Score**: {baseline_avg:.3f}\n")
        f.write(f"- **Optimized Average Score**: {optimized_avg:.3f}\n")
        f.write(f"- **Improvement**: {optimized_avg - baseline_avg:+.3f}\n\n")

        f.write("## Detailed Results\n\n")

        for i, (baseline, optimized) in enumerate(zip(baseline_results, optimized_results), 1):
            f.write(f"### Test Example {i}\n\n")
            f.write(f"**Task**: {baseline['task']}\n\n")
            f.write(f"**Difficulty**: {baseline['difficulty']}\n\n")

            f.write("#### Baseline Agent\n\n")
            f.write(f"- **Score**: {baseline['score']:.2f} ({baseline['raw_score']}/10)\n")
            f.write(f"- **Explanation**: {baseline['explanation']}\n")
            f.write(f"- **Report Length**: {len(baseline['report'])} chars\n\n")

            f.write("#### Optimized Agent\n\n")
            f.write(f"- **Score**: {optimized['score']:.2f} ({optimized['raw_score']}/10)\n")
            f.write(f"- **Explanation**: {optimized['explanation']}\n")
            f.write(f"- **Improvement Suggestions**: {optimized['improvement_suggestions']}\n")
            f.write(f"- **Report Length**: {len(optimized['report'])} chars\n\n")

            f.write(f"**Score Difference**: {optimized['score'] - baseline['score']:+.2f}\n\n")
            f.write("---\n\n")

    print(f"📄 Evaluation report saved: {report_path}")
    return report_path


def main(seed=42):
    """
    Main evaluation function.

    Args:
        seed: Random seed (default: 42)
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"{'=' * 80}")
    print(f"FILE EXPLORATION AGENT - TEST SET EVALUATION")
    print(f"{'=' * 80}")
    print(f"Seed: {seed}")
    print(f"Timestamp: {timestamp}")
    print(f"{'=' * 80}\n")

    # Load test dataset
    print("=📚 Loading test dataset...")
    test_examples = load_file_exploration_dataset(dataset_type="test", random_seed=seed)

    # LM configuration
    print("\n=⚙ Configuring models...")
    fast_lm = configure_lm(FAST_MODEL, temperature=0.0, max_tokens=4096)
    eval_lm = configure_lm(EVAL_MODEL, temperature=0.0, max_tokens=4096)

    # Configure DSPy with inference LM (for agent execution)
    dspy.configure(lm=fast_lm)
    print(f"  Agent execution LM: {FAST_MODEL}")
    print(f"  Evaluation LM: {EVAL_MODEL}")

    # Create LLM as a Judge metric
    print("\n=⚖️ Creating LLM as a Judge evaluation metric...")
    llm_judge_metric = create_llm_judge_metric(eval_lm)

    # Evaluate baseline agent
    print("\n=🔍 Evaluating BASELINE agent...")
    baseline_agent = FileExplorationAgent(max_iters=10, verbose=False)
    baseline_avg, baseline_results = evaluate_agent_detailed(
        baseline_agent,
        test_examples,
        llm_judge_metric,
        eval_lm,
        agent_name="BASELINE"
    )

    # Check if optimized model exists
    if not os.path.exists(GEPA_OPTIMIZED_MODEL_LATEST):
        print(f"\n⚠️ Optimized model not found: {GEPA_OPTIMIZED_MODEL_LATEST}")
        print("Run agent_optimization_gepa.py first to create optimized model.")
        print(f"\nBaseline average score: {baseline_avg:.3f}")
        return

    # Load and evaluate optimized agent
    print(f"\n=📂 Loading optimized agent: {GEPA_OPTIMIZED_MODEL_LATEST}")
    optimized_agent = FileExplorationAgent(max_iters=10, verbose=False)
    optimized_agent.load(GEPA_OPTIMIZED_MODEL_LATEST)

    print("\n=🚀 Evaluating OPTIMIZED agent...")
    optimized_avg, optimized_results = evaluate_agent_detailed(
        optimized_agent,
        test_examples,
        llm_judge_metric,
        eval_lm,
        agent_name="OPTIMIZED"
    )

    # Final comparison
    print(f"\n{'=' * 80}")
    print(f"FINAL COMPARISON")
    print(f"{'=' * 80}")
    print(f"Baseline Average Score:  {baseline_avg:.3f}")
    print(f"Optimized Average Score: {optimized_avg:.3f}")
    print(f"Improvement:             {optimized_avg - baseline_avg:+.3f}")
    print(f"{'=' * 80}\n")

    # Save detailed report
    report_path = save_evaluation_report(
        baseline_results,
        optimized_results,
        baseline_avg,
        optimized_avg,
        timestamp
    )

    print(f"\n✅ Evaluation complete!")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File Exploration Agent Test Set Evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    print(f"🎲 Seed value: {args.seed}")
    main(seed=args.seed)
