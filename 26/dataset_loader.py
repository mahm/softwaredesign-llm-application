"""
JQaRAデータセットのロード機能
"""

import numpy as np
import dspy # type: ignore
from datasets import load_dataset # type: ignore
from collections import defaultdict


def load_jqara_dataset(num_questions: int = 30, max_passages_per_question: int = 10, dataset_split: str = 'dev'):
    """JQaRAデータセットの読み込みと前処理

    Args:
        num_questions: 読み込む質問数（デフォルト30）
        max_passages_per_question: 各質問から使用する最大パッセージ数
        dataset_split: 使用するデータセット分割 ('dev' or 'test')

    Returns:
        examples: DSPy用の質問・回答ペアのリスト
        corpus_texts: 検索用のパッセージコーパス

    Note:
        - devセット: 1質問=50パッセージ（学習・検証用、1,737質問）
        - testセット: 1質問=100パッセージ（評価用、3,334質問）
    """
    # データセットに応じてパッセージ数を設定
    passages_per_question = 50 if dataset_split == 'dev' else 100

    # 必要なレコード数を計算
    num_records = num_questions * passages_per_question
    print(f"📚 JQaRAデータセット({dataset_split})を読み込み中...")

    # データセット読み込み
    ds = load_dataset("hotchpotch/JQaRA", split=f"{dataset_split}[:{num_records}]")

    # 質問ごとにデータを整理
    qid_to_question = {}
    qid_to_answer = {}
    qid_to_passages = defaultdict(list)
    qid_to_labels = defaultdict(list)

    for ex in ds:
        qid = ex["q_id"]
        qid_to_question[qid] = ex["question"]

        # 正解は各行で同じなので先に1つ記録
        if qid not in qid_to_answer and ex["answers"]:
            qid_to_answer[qid] = ex["answers"][0]

        # パッセージとラベルを記録
        qid_to_passages[qid].append(ex["text"])
        qid_to_labels[qid].append(ex.get("label", 0))  # label: 1=正解, 0=不正解

    # 各質問につきExampleを作成
    examples = []
    corpus_texts = []
    correct_counts = []  # 各質問の正解パッセージ数を記録

    for qid in qid_to_question:
        question = qid_to_question[qid]
        answer = qid_to_answer.get(qid, "")
        if not answer:
            continue

        ex = dspy.Example(
            question=question,
            answer=answer
        ).with_inputs("question")
        examples.append(ex)

        # numpyで効率的にパッセージを選択
        passages = np.array(qid_to_passages[qid])
        labels = np.array(qid_to_labels[qid])

        # 正解・不正解のインデックスを取得
        correct_indices = np.where(labels == 1)[0]
        incorrect_indices = np.where(labels == 0)[0]

        # 正解パッセージ数を記録
        correct_counts.append(len(correct_indices))

        # 全ての正解パッセージを選択
        selected_passages = passages[correct_indices].tolist()

        # 残りの枠を不正解パッセージで埋める
        remaining_slots = max_passages_per_question - len(selected_passages)
        if remaining_slots > 0 and len(incorrect_indices) > 0:
            # ランダムに不正解パッセージを選択
            sample_size = min(remaining_slots, len(incorrect_indices))
            sampled_incorrect = np.random.choice(incorrect_indices, sample_size, replace=False)
            selected_passages.extend(passages[sampled_incorrect].tolist())

        corpus_texts.extend(selected_passages)

    # 重複除去
    corpus_texts = list(dict.fromkeys(corpus_texts))

    # 統計情報を表示
    if correct_counts:
        print(f"  質問数: {len(examples)}")
        print(f"  正解パッセージ数: 平均{np.mean(correct_counts):.1f}個 (最小{np.min(correct_counts)}個, 最大{np.max(correct_counts)}個)")
        print(f"  コーパス文書数: {len(corpus_texts)}")

    return examples, corpus_texts