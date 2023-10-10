from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import chainlit as cl

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

それでは始めて下さい！
----------------
{{summaries}}"""


# チャットセッション開始時に実行
@cl.on_chat_start
def chat_start() -> None:
    # 会話履歴を保持するメモリを作成
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question',
        output_key='answer',
        return_messages=True
    )
    # チャットプロンプトを作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("human", "{question}")
    ])
    # データ抽出元のベクトルデータベースの設定
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(
        persist_directory="./data",
        collection_name="events_2023",
        embedding_function=embeddings
    )
    # データソースから関連情報を抽出して返すチェインを定義
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0.0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    # ユーザーセッションにチェインを保存
    cl.user_session.set("chain", chain)


# ユーザーメッセージ受信時に実行
@cl.on_message
async def main(message: str) -> None:
    # ユーザーセッションからチェインを取得
    chain = cl.user_session.get("chain")

    # チェインにユーザーメッセージを渡して回答を取得
    response = await chain.acall(message)

    # 回答と関連情報を取得
    answer = response["answer"]
    sources = response["source_documents"]

    # 関連情報がある場合は、関連情報を表示
    if len(sources) > 0:
        contents = [source.page_content for source in sources]
        await cl.Message(author="relevant", content="\n\n".join(contents), indent=1).send()

    # チャット上にChatGPTからの返信を表示
    await cl.Message(content=answer).send()
