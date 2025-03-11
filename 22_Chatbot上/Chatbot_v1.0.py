# 设置OpenAI和SERPAPI的API密钥
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需的库和模块
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

# 创建一个聊天模型的实例
chat = ChatOpenAI()

# 创建一个消息列表
messages = [
    SystemMessage(content="你是一个花卉行家。"),
    HumanMessage(content="朋友喜欢淡雅的颜色，她的婚礼我选择什么花？")
]

# 使用聊天模型获取响应
response = chat(messages)
print(response)



