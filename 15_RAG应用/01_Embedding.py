from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
# 初始化Embedding类
# 更新导入语句
from langchain_openai import OpenAIEmbeddings  # 使用新的导入路径
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Embed文本
embeddings = embeddings_model.embed_documents(
    [
        "您好，有什么需要帮忙的吗？",
        "哦，你好！昨天我订的花几天送达",
        "请您提供一些订单号？",
        "12345678",
    ]
)
print(len(embeddings), len(embeddings[0]))

# Embed查询
embedded_query = embeddings_model.embed_query("刚才对话中的订单号是多少?")
print(embedded_query[:3])