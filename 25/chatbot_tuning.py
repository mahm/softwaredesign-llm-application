"""枝豆の妖精スタイル チャットボット チューニング"""

import dspy
import os
from dotenv import load_dotenv
from datasets import load_dataset
from chatbot_module import EdamameFairyBot
import mlflow
import mlflow.dspy as mlflow_dspy

load_dotenv()

# MLflowの設定
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")
MLFLOW_TRACKING_URI = f"http://localhost:{MLFLOW_PORT}"
MLFLOW_EXPERIMENT_NAME = "DSPy-EdamameFairy-Optimization"
MLFLOW_RUN_NAME = "miprov2_optimization"

# 最適化されたモジュールの保存先
OPTIMIAZED_MODEL_PATH = "artifact/edamame_fairy_model.json"


def create_style_metric(eval_lm):
    """スタイル評価関数を作成"""
    
    class StyleEvaluation(dspy.Signature):
        """応答のスタイルを評価"""
        response = dspy.InputField(desc="評価対象の応答")
        criteria = dspy.InputField(desc="評価基準")
        score = dspy.OutputField(desc="スコア（0-10）", format=int)
        explanation = dspy.OutputField(desc="評価理由")
    
    evaluator = dspy.ChainOfThought(StyleEvaluation)
    
    def llm_style_metric(_, prediction, __=None):
        """枝豆の妖精スタイルを評価"""
        criteria = """
        以下の基準で0-10点で評価してください:
        1. 語尾に「のだ」「なのだ」を適切に使っているか（3点）
           - 過度な使用（のだのだ等）は減点
           - 自然な日本語として成立しているか
           - 「なのだよ」「なのだね」といった語尾は不自然のため減点
        2. 一人称を使う際は「ボク」を使っているか（2点）
        3. 親しみやすく可愛らしい口調か（3点）
        4. 日本語として自然で読みやすいか（2点）
           - 不自然な繰り返しがないか
           - 文法的に正しいか
        """
        
        # 評価用LMを使用して応答を評価
        with dspy.context(lm=eval_lm):
            eval_result = evaluator(
                response=prediction.response,
                criteria=criteria
            )
        
        # スコアを0-1の範囲に正規化
        score = min(10, max(0, float(eval_result.score))) / 10.0
        return score
    
    return llm_style_metric


def optimize_with_miprov2(trainset, eval_lm, chat_lm):
    """MIPROv2を使用してチャットボットを最適化"""
    
    # データセットをtrain:val = 8:2 の割合で分割
    total_examples = len(trainset)
    train_size = int(total_examples * 0.8)  # 全体の80%を学習用に
    
    # DSPy Exampleのリストを分割
    train_data = trainset[:train_size]      # インデックス0からtrain_sizeまで（学習用）
    evaluation_data = trainset[train_size:] # train_sizeから最後まで（評価用）
    
    # 分割結果の確認と表示
    print("🌱 最適化開始")
    print(f"  総データ数: {total_examples}")
    print(f"  学習用データ: {len(train_data)} ({len(train_data)/total_examples:.1%})")
    print(f"  評価用データ: {len(evaluation_data)} ({len(evaluation_data)/total_examples:.1%})")

    # 最適化対象のチャットボットモジュールを初期化
    chatbot = EdamameFairyBot()
    
    # スタイル評価関数を作成（評価用LMを使用）
    llm_style_metric = create_style_metric(eval_lm)
    
    # DSPyのグローバルLM設定（チャット推論用）
    dspy.configure(lm=chat_lm)
    
    # MIPROv2オプティマイザの設定
    optimizer = dspy.MIPROv2(
        metric=llm_style_metric,  # 評価関数
        prompt_model=eval_lm,     # プロンプト最適化用のLM
        auto="light",             # 最適化モード（light, medium, heavyから選択）
        max_bootstrapped_demos=2,
        max_labeled_demos=1,
    )

    # MLflowの設定
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # MLflowサーバのURL
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME) # MLflowの実験名

    # MLflow DSPyの自動ログ設定
    mlflow_dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True
    )
    
    # MLflowで実行過程をトレース
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        # MIPROv2によるモジュール最適化の実行
        # train_dataを使用してプロンプトと例を自動調整
        optimized_chatbot = optimizer.compile(
            chatbot,
            trainset=train_data,
            minibatch_size=20
        )

        # 評価データでモデルの性能を評価
        eval_score = 0
        for example in evaluation_data:
            # 最適化されたモデルで推論を実行
            prediction = optimized_chatbot(query=example.query, history=example.history)
            # スタイルスコアを計算
            eval_score += llm_style_metric(example, prediction)

        # 平均評価スコアを計算
        avg_eval_score = eval_score / len(evaluation_data)

        # MLflowにメトリクスを記録
        mlflow.log_metric("last_eval_score", avg_eval_score)

        print(f"📊 評価スコア: {avg_eval_score:.3f}")

    return optimized_chatbot


def main():
    """メイン実行関数"""
    
    # 評価用LLMの設定
    eval_lm = dspy.LM(
        model="openai/gpt-4.1-mini",
        temperature=0.0,
        max_tokens=4096
    )

    # チャット推論用LLMの設定
    chat_lm = dspy.LM(
        model="openai/gpt-4.1-nano",
        temperature=0.0,
        max_tokens=1000
    )
        
    # 日本語データセットの読み込み（ずんだもんスタイルの質問応答データ）
    dataset = load_dataset("takaaki-inada/databricks-dolly-15k-ja-zundamon")

    # 使用するデータセットの件数
    train_num = 30
    
    # データセットからDSPy形式のExampleオブジェクトを作成
    # - query: 質問文
    # - history: 会話履歴（今回は空リスト）
    # - response: 期待される応答（学習用）
    trainset = [
        dspy.Example(
            query=item['instruction'],
            history=[],
            response=item['output']
        ).with_inputs("query", "history")  # 入力フィールドを指定
        for item in list(dataset['train'])[:train_num]  # 最初の30件を使用
    ]
    
    # MIPROv2を使用してチャットボットを最適化
    optimized_bot = optimize_with_miprov2(trainset, eval_lm, chat_lm)
    
    # 最適化されたモデルをファイルに保存
    optimized_bot.save(OPTIMIAZED_MODEL_PATH)
    print(f"✅ モデルを保存しました: {OPTIMIAZED_MODEL_PATH}")

    # 保存したモデルを読み込んでテスト
    test_bot = EdamameFairyBot()
    test_bot.load(OPTIMIAZED_MODEL_PATH)

    # テスト用のLM設定（推論用）
    dspy.configure(lm=chat_lm)
    
    # テスト用のクエリ（様々なタイプの質問）
    test_queries = [
        "こんにちは！",
        "枝豆って美味しいよね",
        "DSPyについて教えて"
    ]
    
    # テスト実行と結果表示
    print("\n🧪 テスト結果:")
    for query in test_queries:
        # 最適化されたボットで応答を生成
        result = test_bot(query=query, history=[])
        print(f"Q: {query}")
        print(f"A: {result.response}\n")


if __name__ == "__main__":
    main()