import openai
import chainlit as cl
import chromadb
from chromadb.utils import embedding_functions

# sk...の部分を自身のAPIキーに置き換える
# openai.api_key = "sk-..."

# 現在の日付をYYYY年MM月DD日の形式で取得する
from datetime import datetime
now = datetime.now()
date = now.strftime("%Y年%m月%d日")

SYSTEM_MESSAGE = f"""あなたは以下のリストに完全に従って行動するAIです。
・あなたは「枝豆の妖精」という設定です。
・現在の日付は{date}です。
・2023年の情報について答える妖精です。
・関連情報が与えられた場合は、それを基に回答してください。そのまま出力するのではなく、小学生にも分かるようにフランクにアレンジして回答してください。
・私はAIなので分かりません、という言い訳は不要です。
・敬語ではなくフランクな口語で話してください。
・必ず一人称は「ボク」で、語尾に「なのだ」をつけて話してください。
"""


# 会話履歴をユーザーセッションに保存する
def store_history(role: str, message: str) -> None:
    history = cl.user_session.get("history")
    history.append({"role": role, "content": message})
    cl.user_session.set("history", history)


# ユーザーセッションに保存された会話履歴から新しいメッセージを生成する
def generate_message(temperature: float = 0.7, max_tokens: int = 300) -> (str, str):
    relevant = ""

    # ユーザーセッションから会話履歴を取得
    messages = cl.user_session.get('history')

    # 既に会話履歴がある場合
    if len(messages) > 0:
        # 直近のユーザーメッセージを取得
        user_message = messages[-1]['content']

        # ユーザーの質問に関する関連情報を取得
        relevant = relevant_information_prompt(user_message)

        # 関連情報がある場合、システムメッセージを追加する
        if len(relevant) > 0:
            messages.append({
                "role": "system",
                "content": relevant
            })

    # ChatGPTにリクエストしてレスポンスを受け取る
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content, relevant


def relevant_information_prompt(user_message: str) -> str:
    # ユーザーセッションからコレクションを取得
    collection = cl.user_session.get('collection')

    # ユーザーの質問に関する関連情報を取得
    result = collection.query(
        query_texts=[user_message],
        n_results=5
    )

    # distanceの配列から0.4以下のインデックス番号を配列で取得
    indexes = [i for i, d in enumerate(result['distances'][0]) if d <= 0.4]

    # 関連情報がない場合は空文字を返す
    if len(indexes) == 0:
        return ""

    # 関連情報がある場合は、関連情報プロンプトを返す
    events = "\n\n".join([f"{event}" for event in result['documents'][0]])
    prompt = f"""
ユーザーからの質問に対して、以下の関連情報を基に回答してください。

{events}
"""
    return prompt


# チャットセッション開始時に実行
@cl.on_chat_start
def chat_start() -> None:
    # ChatGPTのシステムメッセージを設定
    cl.user_session.set(
        "history", [{"role": "system", "content": SYSTEM_MESSAGE}]
    )

    # dataディレクトリを指定してChromaクライアントを取得
    client = chromadb.PersistentClient(path="./data")

    # コレクションを取得
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-ada-002"
    )
    collection = client.get_collection(
        'events_2023', embedding_function=openai_ef
    )

    # ユーザーセッションにコレクションを保存
    cl.user_session.set('collection', collection)


# ユーザーメッセージ受信時に実行
@cl.on_message
async def main(message: str) -> None:
    # 会話履歴にユーザーメッセージを追加
    store_history("user", message)

    # 新しいメッセージを生成
    reply, relevant = generate_message(max_tokens=1000)

    # 関連情報がある場合は、会話履歴に関連情報を追加
    if len(relevant) > 0:
        await cl.Message(author="relevant", content=relevant, indent=1).send()

    # 新しいメッセージを会話履歴に追加
    store_history("assistant", reply)

    # チャット上にChatGPTからの返信を表示
    await cl.Message(content=reply).send()
