from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()  # 加载 .env 文件中的环境变量

# 系统模板的构建
template: str = "你是一位专业顾问，负责为专注于{product}的公司起名。"
system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(template)

# 人类模板的构建
human_template: str = "公司主打产品是{product_detail}。"
human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(human_template)

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 格式化提示消息生成提示
messags: List[BaseMessage] = prompt_template.format_prompt(
    product="鲜花装饰", 
    product_detail="创新的鲜花设计。"
).to_messages()

# 调用模型生成结果
chat: ChatOpenAI = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)
result = chat.invoke(messags)
print(result.content)