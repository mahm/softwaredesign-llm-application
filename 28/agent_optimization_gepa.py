# -*- coding: utf-8 -*-
"""
File Exploration Agent Optimization Script (GEPA)

This script optimizes the file exploration agent using GEPA (Genetic-Pareto) optimization.
"""

import os
import sys
import random
import argparse
import logging
import dspy
from datetime import datetime

from config import configure_lm, SMART_MODEL, FAST_MODEL
from agent_module import FileExplorationAgent
from dataset_loader import load_file_exploration_dataset

# Optimized model save path (symlink to latest)
GEPA_OPTIMIZED_MODEL_LATEST = "artifact/agent_gepa_optimized_latest.json"


class Tee:
    """Class to output to both console and file"""

    def __init__(self, file_path: str, original_stdout):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = original_stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class ReportEvaluation(dspy.Signature):
    """
    ファイル探索レポートを評価します。

    criteriaに記載された評価基準に厳密に従って評価します。
    このSignatureは評価の「型」のみを定義し、
    具体的な評価ロジックは全てcriteriaに委譲します。
    """

    task: str = dspy.InputField(
        desc="ファイル探索タスクの説明。エージェントに与えられた指示。"
    )

    report: str = dspy.InputField(
        desc="エージェントが生成したレポートの全文。"
    )

    criteria: str = dspy.InputField(
        desc="""評価基準の完全な記述。

このフィールドには以下が完全に明示されています：
- スコアリング方法（0-10点の配分）
- 必須ファイルの完全リスト（config.py, rag_optimization_gepa.pyなど）
- オプションファイルの完全リスト（README.md: +0.5点など）
- 必須要素の完全リスト（各要素の配点含む）
- 情報統合の評価基準

曖昧な表現（「等」「など」「主要な」）は一切含まれません。
このcriteriaに記載された基準に厳密に従ってください。"""
    )

    score: int = dspy.OutputField(
        desc="criteriaに基づいて算出された総合スコア（0-10の整数）。"
    )

    explanation: str = dspy.OutputField(
        desc="""評価理由の詳細（200-400文字）。

以下を含めてください：
1. ファイル読み取り評価: どのファイルを読んだか、criteriaの必須ファイルと照合
2. 必須要素評価: criteriaの必須要素リストと照合、含まれた要素/欠落要素
3. 情報統合評価: 複数ファイル間の関係性説明の質
4. スコア内訳: 各項目で何点獲得したか"""
    )

    improvement_suggestions: str = dspy.OutputField(
        desc="""GEPAリフレクション用の改善提案（150-300文字）。

具体的で実行可能な提案を記述してください：
- "taskにファイル名が含まれていたら、まずそのファイルを読むべき"
- "import文を見つけたら、そのインポート元ファイルも読むべき"
- "変数定義を見つけたら、その変数の使用箇所も探すべき"

抽象的な提案（「もっと詳しく」など）は避けてください。"""
    )


def create_llm_judge_metric(eval_lm):
    """
    LLM as a Judge評価メトリックを作成します。

    Args:
        eval_lm: 評価用LM (gpt-4.1-mini推奨)

    Returns:
        評価関数（gold, pred, trace=None → float score）
    """
    evaluator = dspy.ChainOfThought(ReportEvaluation)

    def llm_judge_metric(gold, pred, trace=None):
        """
        LLM as a Judgeによる評価。

        Args:
            gold: Gold標準データ（criteriaを含む）
            pred: 予測結果（reportを含む）
            trace: 実行トレース（optional）

        Returns:
            float: 0.0-1.0のスコア
        """
        # レポートが存在しない場合は0点
        if not hasattr(pred, 'report') or not pred.report:
            return 0.0

        # criteriaが存在しない場合はエラー
        if not hasattr(gold, 'criteria') or not gold.criteria:
            raise ValueError("Gold example must have 'criteria' field for LLM as a Judge evaluation")

        # LLM as a Judgeで評価
        with dspy.context(lm=eval_lm):
            eval_result = evaluator(
                task=gold.task,
                report=pred.report,
                criteria=gold.criteria
            )

        # スコアを0-10から0-1に正規化
        raw_score = eval_result.score
        try:
            score = float(raw_score)
            score = min(10.0, max(0.0, score)) / 10.0
        except (ValueError, TypeError):
            score = 0.0

        return score

    return llm_judge_metric


def create_gepa_llm_judge_metric(eval_lm):
    """
    GEPA最適化用のLLM as a Judge評価メトリックを作成します。

    GEPAが要求するScoreWithFeedback形式（score + feedback + improvement_suggestions）を返します。

    Args:
        eval_lm: 評価用LM (gpt-4.1-mini推奨)

    Returns:
        評価関数（gold, pred, trace=None, pred_name=None, pred_trace=None → dspy.Prediction）
    """
    evaluator = dspy.ChainOfThought(ReportEvaluation)

    def gepa_llm_judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """
        GEPA用LLM as a Judge評価。

        Args:
            gold: Gold標準データ（criteriaを含む）
            pred: 予測結果（reportを含む）
            trace: Program execution trace (optional)
            pred_name: Name of specific predictor being optimized (optional)
            pred_trace: Execution trace of specific predictor (optional)

        Returns:
            dspy.Prediction: ScoreWithFeedback type (score, feedback, improvement_suggestions)
        """
        # レポートが存在しない場合
        if not hasattr(pred, 'report') or not pred.report:
            return dspy.Prediction(
                score=0.0,
                feedback="❌ No report generated",
                improvement_suggestions="レポートを生成するために、ls_directory、read_file、write_fileツールを適切に使用してください。"
            )

        # criteriaが存在しない場合はエラー
        if not hasattr(gold, 'criteria') or not gold.criteria:
            raise ValueError("Gold example must have 'criteria' field for LLM as a Judge evaluation")

        # LLM as a Judgeで評価
        with dspy.context(lm=eval_lm):
            eval_result = evaluator(
                task=gold.task,
                report=pred.report,
                criteria=gold.criteria
            )

        # スコアを0-10から0-1に正規化
        raw_score = eval_result.score
        try:
            score = float(raw_score)
            score = min(10.0, max(0.0, score)) / 10.0
        except (ValueError, TypeError):
            score = 0.0

        # フィードバック（explanationから簡潔版を生成）
        feedback = f"Score: {raw_score}/10"
        if hasattr(eval_result, 'explanation') and eval_result.explanation:
            # explanationの最初の100文字を抽出
            explanation_short = eval_result.explanation[:100]
            if len(eval_result.explanation) > 100:
                explanation_short += "..."
            feedback += f" | {explanation_short}"

        # 改善提案（GEPA reflection用）
        improvement_suggestions = ""
        if hasattr(eval_result, 'improvement_suggestions') and eval_result.improvement_suggestions:
            improvement_suggestions = eval_result.improvement_suggestions

        # predictor名を追加
        if pred_name:
            feedback += f" | [{pred_name}]"

        # GEPA expects ScoreWithFeedback type
        return dspy.Prediction(
            score=score,
            feedback=feedback,
            improvement_suggestions=improvement_suggestions,
            # 詳細情報を保持（ログ用）
            explanation=eval_result.explanation if hasattr(eval_result, 'explanation') else "",
            raw_score=raw_score
        )

    return gepa_llm_judge_metric


def gepa_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA metric function with score + textual feedback.

    Args:
        gold: Gold standard data
        pred: Prediction result
        trace: Program execution trace (optional)
        pred_name: Name of specific predictor being optimized (optional)
        pred_trace: Execution trace of specific predictor (optional)

    Returns:
        dspy.Prediction: ScoreWithFeedback type (with score and feedback fields)
    """
    # Calculate base score
    score = file_exploration_metric(gold, pred, trace)

    # Generate feedback
    feedback_parts = []

    # Evaluate report quality
    if not hasattr(pred, 'report') or not pred.report:
        feedback_parts.append(" No report generated")
    else:
        report = pred.report.strip()
        report_len = len(report)

        if report_len >= 100:
            feedback_parts.append(f" Substantial report ({report_len} chars)")
        else:
            feedback_parts.append(f"� Short report ({report_len} chars)")

        # Check for error messages
        if any(err in report.lower() for err in ['error:', 'failed', 'could not']):
            feedback_parts.append("� Contains error messages")

        # Check structure
        lines = len(report.split('\n'))
        if lines >= 5:
            feedback_parts.append(f" Well-structured ({lines} lines)")
        elif lines >= 2:
            feedback_parts.append(f"� Basic structure ({lines} lines)")

    # Task-specific feedback
    if hasattr(gold, 'task'):
        task_lower = gold.task.lower()
        if 'list' in task_lower or 'find' in task_lower:
            if hasattr(pred, 'report') and pred.report:
                # Check if report contains file listings
                if any(ext in pred.report for ext in ['.py', '.json', '.md', '.txt', 'FILE', 'DIR']):
                    feedback_parts.append(" Contains file information")

    # Add predictor name if specified
    if pred_name:
        feedback_parts.append(f"[{pred_name}]")

    # Generate feedback string
    feedback = " | ".join(feedback_parts)

    # GEPA expects ScoreWithFeedback type (dspy.Prediction)
    return dspy.Prediction(
        score=score,
        feedback=feedback
    )


def log_metric_evaluation(gold, pred, trace, pred_name, pred_trace, result):
    """
    Log metric evaluation.

    Args:
        gold: Gold standard data
        pred: Prediction result
        trace: Program execution trace
        pred_name: Predictor name
        pred_trace: Predictor execution trace
        result: Metric calculation result (dspy.Prediction)
    """
    logger = logging.getLogger("gepa_optimization")

    # Log input arguments
    logger.info(f"gold = {repr(gold)}")
    logger.info(f"pred = {repr(pred)}")
    logger.info(f"trace = {repr(trace) if trace is not None else 'None'}")
    logger.info(f"pred_name = {repr(pred_name) if pred_name is not None else 'None'}")
    logger.info(f"pred_trace = {repr(pred_trace) if pred_trace is not None else 'None'}")

    logger.info("-" * 80)

    # Log calculation results
    logger.info(f"score = {result.score}")
    logger.info(f"feedback = {result.feedback}")
    logger.info("=" * 80)


def gepa_metric_with_feedback_logged(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA metric function with logging.

    Args:
        gold: Gold standard data
        pred: Prediction result
        trace: Program execution trace (optional)
        pred_name: Name of specific predictor being optimized (optional)
        pred_trace: Execution trace of specific predictor (optional)

    Returns:
        dspy.Prediction: ScoreWithFeedback type (with score and feedback fields)
    """
    result = gepa_metric_with_feedback(gold, pred, trace, pred_name, pred_trace)
    log_metric_evaluation(gold, pred, trace, pred_name, pred_trace, result)

    return result


def setup_logging(timestamp: str) -> tuple:
    """
    Setup logging environment.

    Args:
        timestamp: Timestamp string (YYYYMMDD_HHMM format)

    Returns:
        tuple: (original_stdout, tee, log_path, stdout_path)
    """
    # Create log directory
    os.makedirs("logs", exist_ok=True)

    # Log file paths
    log_filename = f"gepa_optimization_{timestamp}.log"
    log_path = os.path.join("logs", log_filename)

    stdout_filename = f"gepa_optimization_{timestamp}_stdout.log"
    stdout_path = os.path.join("logs", stdout_filename)

    # Logger configuration
    logger = logging.getLogger("gepa_optimization")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout
    original_stdout = sys.stdout
    tee = Tee(stdout_path, original_stdout)
    sys.stdout = tee

    print(f"=� Log file: {log_path}")
    print(f"=� Stdout log: {stdout_path}")

    return original_stdout, tee, log_path, stdout_path


def cleanup_logging(original_stdout, tee: Tee, log_path: str, stdout_path: str) -> None:
    """
    Cleanup logging environment.

    Args:
        original_stdout: Original stdout
        tee: Tee object
        log_path: Log file path
        stdout_path: Stdout log path
    """
    # Cleanup logger
    logger = logging.getLogger("gepa_optimization")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Restore stdout
    sys.stdout = original_stdout
    tee.close()

    # Display file paths
    print("\n Optimization complete!")
    print(f"=� Detailed log: {log_path}")
    print(f"=� Standard output: {stdout_path}")


def main(seed=42, dataset="train"):
    """
    Main execution function.

    Args:
        seed: Random seed (default: 42)
    """
    # Generate timestamp (shared by log and model filenames)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Setup logging environment
    original_stdout, tee, log_path, stdout_path = setup_logging(timestamp)

    try:
        # Load dataset
        print("=� Loading file exploration dataset...")
        train_examples = load_file_exploration_dataset(dataset_type=dataset, random_seed=seed)
        # Note: test set is loaded separately in agent_evaluation.py
        # For GEPA optimization, we use train set only (no val set)

        # LM configuration
        print("\n=' Configuring models...")
        # GEPA reflection LM (high temperature)
        reflection_lm = configure_lm(SMART_MODEL, temperature=1.0, max_tokens=8192)
        # Inference LM (fast model)
        fast_lm = configure_lm(FAST_MODEL, temperature=0.0, max_tokens=4096)
        # Evaluation LM (for LLM as a Judge)
        from config import EVAL_MODEL
        eval_lm = configure_lm(EVAL_MODEL, temperature=0.0, max_tokens=4096)

        # Configure DSPy with default LM
        dspy.configure(lm=fast_lm)

        # Create LLM as a Judge metrics
        print("\n=\u2696\ufe0f Creating LLM as a Judge evaluation metrics...")
        llm_judge_metric = create_llm_judge_metric(eval_lm)
        gepa_llm_metric = create_gepa_llm_judge_metric(eval_lm)

        # Baseline evaluation
        print("\n=� Evaluating baseline (train set)...")
        baseline_agent = FileExplorationAgent(max_iters=10, verbose=False)

        baseline_scores = []
        for ex in train_examples[:3]:  # Use first 3 examples for quick baseline check
            pred = baseline_agent(task=ex.task, working_directory=ex.working_directory)
            score = llm_judge_metric(ex, pred)
            baseline_scores.append(score)

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        print(f"  Baseline average score: {baseline_avg:.3f} (on {len(baseline_scores)} examples)")

        # GEPA optimization
        print("\n=� Starting GEPA optimization...")

        # Target agent for optimization
        agent = FileExplorationAgent(max_iters=10, verbose=False)

        # GEPA configuration
        optimizer = dspy.GEPA(
            metric=gepa_llm_metric,  # LLM as a Judge metric with feedback
            auto="light",  # Optimization intensity
            reflection_lm=reflection_lm,  # LM for reflection (strong model recommended)
        )

        # Execute optimization
        optimized_agent = optimizer.compile(
            agent,
            trainset=train_examples,
        )

        # Post-optimization evaluation
        print("\n=� Evaluating optimized agent (train set)...")
        opt_scores = []
        for ex in train_examples[:3]:  # Use first 3 examples for quick check
            pred = optimized_agent(task=ex.task, working_directory=ex.working_directory)
            score = llm_judge_metric(ex, pred)
            opt_scores.append(score)

        opt_avg = sum(opt_scores) / len(opt_scores) if opt_scores else 0.0
        print(f"  [Baseline] Avg score: {baseline_avg:.3f} (on {len(baseline_scores)} examples)")
        print(f"  [GEPA Optimized] Avg score: {opt_avg:.3f} (on {len(opt_scores)} examples)")
        print(f"  Improvement: {opt_avg - baseline_avg:+.3f}")

        # Generate filename with score (use validation score, reuse timestamp)
        score_percent = int(opt_avg * 100)
        score_str = f"score{score_percent:03d}"
        model_filename = f"agent_gepa_optimized_{timestamp}_{score_str}.json"
        model_path = os.path.join("artifact", model_filename)

        # Save model
        os.makedirs("artifact", exist_ok=True)
        optimized_agent.save(model_path)
        print(f"\n=� Saved optimized model: {model_path}")

        # Create symlink to latest version
        if os.path.exists(GEPA_OPTIMIZED_MODEL_LATEST):
            os.remove(GEPA_OPTIMIZED_MODEL_LATEST)
        os.symlink(model_filename, GEPA_OPTIMIZED_MODEL_LATEST)

        print(f"  � Latest link: {GEPA_OPTIMIZED_MODEL_LATEST}")

    finally:
        # Cleanup logging environment
        cleanup_logging(original_stdout, tee, log_path, stdout_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File Exploration Agent Optimization (GEPA)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--dataset', type=str, default='train',
                       choices=['train', 'mini_test'],
                       help='Dataset to use: train (10 examples) or mini_test (3 examples)')
    args = parser.parse_args()

    print(f"<1 Seed value: {args.seed}")
    print(">� Optimization method: GEPA (Genetic-Pareto)")
    print(f"📊 Dataset: {args.dataset}")
    main(seed=args.seed, dataset=args.dataset)
