import os
import chainlit as cl
from conversational_agent import ConversationalAgent
from pathlib import Path

# 作業用ディレクトリの設定
WORKING_DIRECTORY = Path('./work')

# sk...の部分を自身のOpenAIのAPIキーに置き換える
# os.environ["OPENAI_API_KEY"] = "sk-..."


# チャットセッション開始時に実行
@cl.on_chat_start
def chat_start() -> None:
    # エージェントを初期化
    agent = ConversationalAgent(working_directory=WORKING_DIRECTORY)

    # ユーザーセッションにエージェントを保存
    cl.user_session.set("agent", agent)


# ユーザーメッセージ受信時に実行
@cl.on_message
async def main(message: cl.Message) -> None:
    # ユーザーセッションからチェインを取得
    agent = cl.user_session.get("agent")

    # エージェントからの返答を表示
    async for step in agent.run(message.content):
        if step["is_final"]:
            await cl.Message(content=step["message"]).send()
        else:
            await cl.Message(author="tool", content=step["message"], indent=1).send()
