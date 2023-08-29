import chainlit as cl


@cl.on_message
async def main(message: str):
    await cl.Message(content=f"ボクは枝豆の妖精なのだ。").send()
