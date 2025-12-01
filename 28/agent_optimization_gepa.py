"""
ファイル探索エージェント最適化スクリプト（GEPA）
"""

import os
import sys
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
このcriteriaに記載された基準に厳密に従ってください。

【重要】必須ファイル未読時の評価ルール:
1. 必須ファイルを全く読んでいない場合（0ファイル）: 総合スコア0点
2. 必須ファイルを一部読んでいる場合: 読んだファイル数に応じて部分点を付与
   - ファイル読み取り点: 読んだファイル数 / 必須ファイル総数 × 配点
   - 必須要素の言及: 読んだファイルに関連する要素のみ評価、未読ファイルの要素は0点
   - 情報統合: 読んだファイルの範囲内で評価
3. ファイルを読まずに推測や一般知識のみで説明している場合は評価しない（ハルシネーション）
4. 読んだファイルの内容に基づく説明は、たとえ一部のファイルのみでも評価対象"""
    )

    trajectory: str = dspy.InputField(
        desc="""エージェントのツール呼び出し履歴（オプショナル）。

フォーマット:
- thought_0, tool_name_0, tool_args_0, observation_0
- thought_1, tool_name_1, tool_args_1, observation_1
- ...

この情報を使って、エージェントが実際にどのファイルを読んだかを確認できます。
reportに証拠が不足している場合でも、trajectoryで確認してください。

例: tool_name_2="read_file", tool_args_2={"file_path": "constants.py"}
→ constants.pyを確実に読んでいる

【重要】trajectoryで確認できることが優先されます。reportの記述が不十分でも、
trajectoryでファイルを読んでいることが確認できれば、ファイル読み取り点は付与してください。
ただし、reportの品質が低い場合はimprovement_suggestionsで指摘してください。"""
    )

    score: int = dspy.OutputField(
        desc="criteriaに基づいて算出された総合スコア（0-10の整数）。"
    )

    explanation: str = dspy.OutputField(
        desc="""評価理由の詳細（200-400文字）。

以下を含めてください：
1. ファイル読み取り評価: trajectoryとreportの両方を確認
   - trajectoryで tool_name_N="read_file" を確認（確実な証拠）
   - reportに具体的な値の引用があるか確認（補助的証拠）
   ⚠️ trajectoryで読んでいることが確認できれば、reportが不十分でもファイル読み取り点を付与
   ⚠️ trajectoryにもreportにも証拠がない場合は「ハルシネーション」と明記
2. 必須要素評価: criteriaの必須要素リストと照合、含まれた要素/欠落要素
3. 情報統合評価: 複数ファイル間の関係性説明の質
4. スコア内訳: 各項目で何点獲得したか"""
    )

    improvement_suggestions: str = dspy.OutputField(
        desc="""GEPAリフレクション用の改善提案（150-300文字）。

具体的で実行可能な提案を記述してください：
- "taskにファイル名が含まれていたら、まずそのファイルを必ず読むべき"
- "ファイルが見つからない場合は、recursive=True と pattern='*.py' で再帰探索すべき"
- "import文を見つけたら、そのインポート元ファイルも読むべき"
- "変数定義を見つけたら、その変数の使用箇所も探すべき"
- "必須ファイル未読の場合は、推測で回答せず必ずファイルを探すべき"

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

        # trajectoryをフォーマットして渡す
        trajectory_str = ""
        if hasattr(pred, 'trajectory') and pred.trajectory:
            # trajectoryを整形（長すぎる場合は切り詰め）
            trajectory_items = []
            for k, v in pred.trajectory.items():
                v_str = str(v)
                # 各観測結果は最大500文字まで
                if len(v_str) > 500:
                    v_str = v_str[:500] + "... (truncated)"
                trajectory_items.append(f"{k}: {v_str}")
            trajectory_str = "\n".join(trajectory_items)

        # LLM as a Judgeで評価
        with dspy.context(lm=eval_lm):
            eval_result = evaluator(
                task=gold.task,
                report=pred.report,
                criteria=gold.criteria,
                trajectory=trajectory_str
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
                feedback="[ERROR] No report generated",
                improvement_suggestions="レポートを生成するために、ls_directory、read_file、write_fileツールを適切に使用してください。"
            )

        # criteriaが存在しない場合はエラー
        if not hasattr(gold, 'criteria') or not gold.criteria:
            raise ValueError("Gold example must have 'criteria' field for LLM as a Judge evaluation")

        # trajectoryをフォーマットして渡す
        trajectory_str = ""
        if hasattr(pred, 'trajectory') and pred.trajectory:
            # trajectoryを整形（長すぎる場合は切り詰め）
            trajectory_items = []
            for k, v in pred.trajectory.items():
                v_str = str(v)
                # 各観測結果は最大500文字まで
                if len(v_str) > 500:
                    v_str = v_str[:500] + "... (truncated)"
                trajectory_items.append(f"{k}: {v_str}")
            trajectory_str = "\n".join(trajectory_items)

        # LLM as a Judgeで評価
        with dspy.context(lm=eval_lm):
            eval_result = evaluator(
                task=gold.task,
                report=pred.report,
                criteria=gold.criteria,
                trajectory=trajectory_str
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



# Removed deprecated functions (2025-11-22):
# - gepa_metric_with_feedback (unused, called undefined file_exploration_metric)
# - log_metric_evaluation (only used by gepa_metric_with_feedback_logged)
# - gepa_metric_with_feedback_logged (unused)
# Replaced by: create_llm_judge_metric() and create_gepa_llm_judge_metric()

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

    print(f"[LOG] Log file: {log_path}")
    print(f"[LOG] Stdout log: {stdout_path}")

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
    print("\n[DONE] Optimization complete!")
    print(f"[LOG] Detailed log: {log_path}")
    print(f"[LOG] Standard output: {stdout_path}")


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
        print("[LOG] Loading file exploration dataset...")
        train_examples = load_file_exploration_dataset(dataset_type=dataset, random_seed=seed)
        # Note: test set is loaded separately in agent_evaluation.py
        # For GEPA optimization, we use train set only (no val set)

        # LM configuration
        print("\n[CONFIG] Configuring models...")
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
        print("\n[EVAL] Creating LLM as a Judge evaluation metrics...")
        llm_judge_metric = create_llm_judge_metric(eval_lm)
        gepa_llm_metric = create_gepa_llm_judge_metric(eval_lm)

        # Baseline evaluation
        print("\n[EVAL] Evaluating baseline (train set)...")
        baseline_agent = FileExplorationAgent(max_iters=10, verbose=False)

        baseline_scores = []
        for ex in train_examples[:3]:  # Use first 3 examples for quick baseline check
            pred = baseline_agent(task=ex.task, working_directory=ex.working_directory)
            score = llm_judge_metric(ex, pred)
            baseline_scores.append(score)

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        print(f"  Baseline average score: {baseline_avg:.3f} (on {len(baseline_scores)} examples)")

        # GEPA optimization
        print("\n[START] Starting GEPA optimization...")

        # Target agent for optimization
        agent = FileExplorationAgent(max_iters=10, verbose=False)

        # GEPA configuration
        optimizer = dspy.GEPA(
            metric=gepa_llm_metric,  # LLM as a Judge metric with feedback
            auto="light",  # Optimization intensity (light=6 candidates)
            reflection_lm=reflection_lm,  # LM for reflection (strong model recommended)
        )

        # Execute optimization
        optimized_agent = optimizer.compile(
            agent,
            trainset=train_examples,
        )

        # Post-optimization evaluation
        print("\n[EVAL] Evaluating optimized agent (train set)...")
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
        print(f"\n[SAVE] Saved optimized model: {model_path}")

        # Create symlink to latest version
        if os.path.exists(GEPA_OPTIMIZED_MODEL_LATEST):
            os.remove(GEPA_OPTIMIZED_MODEL_LATEST)
        os.symlink(model_filename, GEPA_OPTIMIZED_MODEL_LATEST)

        print(f"  [LINK] Latest link: {GEPA_OPTIMIZED_MODEL_LATEST}")

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

    print(f"[SEED] Seed value: {args.seed}")
    print(f"[METHOD] Optimization method: GEPA (Genetic-Pareto)")
    print(f"[DATA] Dataset: {args.dataset}")
    main(seed=args.seed, dataset=args.dataset)
