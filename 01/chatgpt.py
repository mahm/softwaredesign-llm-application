import openai

# sk_...の部分は手元に控えているAPIキーの値に置き換えて下さい
openai.api_key = "sk_..."

messages = [
    {"role": "system", "content": "あなたは枝豆の妖精です。一人称は「ボク」で、語尾に「なのだ」をつけて話すことが特徴です。"},
    {"role": "user", "content": "こんにちは！"}
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.7,
    max_tokens=300,
)

print(response.choices[0].message.content)
