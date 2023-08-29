import openai
import chainlit as cl

# sk...の部分を自身のAPIキーに置き換える
openai.api_key = "sk-..."


# 会話履歴をユーザーセッションに保存する
def store_history(role, message):
    history = cl.user_session.get("history")
    history.append({"role": role, "content": message})
    cl.user_session.set("history", history)


# ユーザーセッションに保存された会話履歴から新しいメッセージを生成する
def generate_message():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=cl.user_session.get("history"),
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content


# チャットセッション開始時に実行
@cl.on_chat_start
def chat_start():
    cl.user_session.set("history", [{
        "role": "system",
        "content": "あなたは枝豆の妖精です。一人称は「ボク」で、語尾に「なのだ」をつけて話すことが特徴です。"
    }])


# ユーザーメッセージ受信時に実行
@cl.on_message
async def main(message: str):
    # 会話履歴にユーザーメッセージを追加
    store_history("user", message)

    # 新しいメッセージを生成
    reply = generate_message()

    # 新しいメッセージを会話履歴に追加
    store_history("assistant", message)

    # チャット上にChatGPTからの返信を表示
    msg = cl.Message(content=reply)
    await msg.send()
