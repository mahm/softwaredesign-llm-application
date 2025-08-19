"""
枝豆の妖精スタイルの対話型チャット
"""

import dspy
from dotenv import load_dotenv
from collections import deque
from chatbot_module import EdamameFairyBot
from chatbot_tuning import OPTIMIAZED_MODEL_PATH

load_dotenv()

def main():
    # チャット用LM設定
    lm = dspy.LM(
        model="openai/gpt-4.1-nano",
        temperature=0.0,
        max_tokens=1000
    )
    
    # DSPy標準のグローバル設定
    dspy.configure(lm=lm)
    
    # モデル読み込み
    print("📂 モデル読み込み中...")
    chatbot = EdamameFairyBot()
    chatbot.load(OPTIMIAZED_MODEL_PATH)
    print("✅ 最適化済みモデルを読み込みました")
    
    # 対話履歴を管理
    history = deque(maxlen=5)
    
    print("\n🌱 枝豆の妖精チャットボット")
    print("（'quit' または 'exit' で終了）")
    print("-" * 50)
    
    while True:
        user_input = input("\nあなた: ")
        
        if user_input.lower() in ['quit', 'exit', '終了']:
            print("\n🌱妖精: バイバイなのだ！")
            break
        
        # 履歴をリスト形式に変換
        history_list = [f"User: {h[0]}\nBot: {h[1]}" for h in history]
        
        # 応答生成
        result = chatbot(query=user_input, history=history_list)
        print(f"🌱妖精: {result.response}")
        
        # 履歴に追加
        history.append((user_input, result.response))

if __name__ == "__main__":
    main()