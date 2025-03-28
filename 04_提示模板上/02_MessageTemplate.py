from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

load_dotenv()  # 加载 .env 文件中的环境变量

# 系统模板的构建
template: str = "你是一位专业顾问，负责为专注于{product}的公司起名。"

# 人类模板的构建
human_template: str = "公司主打产品是{product_detail}。"

# 使用 format 方法传入变量值
product = "智能家居"
product_detail = "智能灯泡"

messages = [
    SystemMessage(content=template.format(product=product)),
    HumanMessage(content=human_template.format(product_detail=product_detail))
]

# 调用模型生成结果
chat: ChatOpenAI = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)
result = chat.invoke(messages)
print(result.content)