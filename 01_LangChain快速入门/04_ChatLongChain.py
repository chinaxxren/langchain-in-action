import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 更新导入语句
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(
    model="gpt-4",
    temperature=0.8,
    max_tokens=60)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]

# 使用 invoke 方法替代直接调用
response = chat.invoke(messages)
print(response.content)