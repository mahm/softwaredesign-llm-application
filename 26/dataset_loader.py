"""
JQaRAデータセットのロード機能
"""

import numpy as np
import pandas as pd # type: ignore
import dspy # type: ignore
from datasets import load_dataset # type: ignore


def load_jqara_dataset(num_questions: int = 30, dataset_split: str = 'dev', random_seed: int = 42):
    """JQaRAデータセットの読み込みと前処理

    Args:
        num_questions: 読み込む質問数（デフォルト30）
        dataset_split: 使用するデータセット分割 ('dev' or 'test')
        random_seed: ランダムシード（再現性のため）

    Returns:
        examples: DSPy用の質問・回答ペアのリスト
                  各Exampleはquestion, answer, positives, negativesフィールドを含む
        corpus_texts: 検索用のパッセージコーパス（全パッセージの混在リスト）

    Note:
        - devセット: 1質問=50パッセージ（学習・検証用、1,737質問）
        - testセット: 1質問=100パッセージ（評価用、3,334質問）
        - 各質問の全パッセージを使用
        - パッセージはtitleとtextを結合
        - positives: 正解を含むパッセージのリスト
        - negatives: 正解を含まないパッセージのリスト
    """
    # ランダムシードを固定（再現性のため）
    np.random.seed(random_seed)

    # データセットに応じてパッセージ数を設定
    passages_per_question = 50 if dataset_split == 'dev' else 100

    # 必要なレコード数を計算
    num_records = num_questions * passages_per_question
    print(f"📚 JQaRAデータセット({dataset_split})を読み込み中...")

    # データセット読み込み
    ds = load_dataset("hotchpotch/JQaRA", split=f"{dataset_split}[:{num_records}]")

    # pandasのDataFrameに変換
    df = ds.to_pandas()

    # パッセージを作成（title + text形式）
    df['passage'] = df['title'] + '\n' + df['text']

    def aggregate_group(group):
        """グループごとに集約処理を行う"""
        # 質問と回答を取得（グループ内では全て同じ）
        question = group['question'].iloc[0]
        answers = group['answers'].iloc[0]

        # 最初の回答のみ使用
        answer = answers[0] if answers else ""

        # labelに基づいてpositivesとnegativesを分離
        mask_positive = group['label'] == 1
        positives = group.loc[mask_positive, 'passage'].tolist()
        negatives = group.loc[~mask_positive, 'passage'].tolist()

        # シャッフル（numpy使用）
        np.random.shuffle(positives)
        np.random.shuffle(negatives)

        return pd.Series({
            'question': question,
            'answer': answer,
            'positives': positives,
            'negatives': negatives,
            'num_positives': len(positives),
            'num_negatives': len(negatives)
        })

    # q_idでグループ化して集約
    result = df.groupby('q_id', as_index=False).apply(
        aggregate_group, include_groups=False
    )

    # 統計情報の計算
    correct_counts = result['num_positives'].values
    print(f"  質問数: {len(result)}")
    print(f"  正解パッセージ数: 平均{np.mean(correct_counts):.1f}個 (最小{np.min(correct_counts)}個, 最大{np.max(correct_counts)}個)")

    # DSPy用のExampleを作成
    examples = []
    corpus_texts = []

    for _, row in result.iterrows():
        if not row['answer']:
            continue

        # 正例と負例を含むExampleを作成
        ex = dspy.Example(
            question=row['question'],
            answer=row['answer'],
            positives=row['positives'],
            negatives=row['negatives']
        ).with_inputs("question")

        examples.append(ex)

        # コーパスに追加（全パッセージをシャッフルして混在）
        all_passages = row['positives'] + row['negatives']
        np.random.shuffle(all_passages)
        corpus_texts.extend(all_passages)

    print(f"  コーパス文書数: {len(corpus_texts)}")

    return examples, corpus_texts