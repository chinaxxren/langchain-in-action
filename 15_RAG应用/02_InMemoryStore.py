#embed_documents() ：

# - 接收一个文本列表作为输入
# - 将每个文本转换为向量表示
# - 自动处理缓存逻辑

from dotenv import load_dotenv
load_dotenv()

# 导入内存存储库
from langchain.storage import InMemoryStore
store = InMemoryStore()

# 更新导入路径
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

import os
# 创建 OpenAIEmbeddings 实例
underlying_embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 创建一个CacheBackedEmbeddings的实例。
# 这将为underlying_embeddings提供缓存功能，嵌入会被存储在上面创建的InMemoryStore中。
# 我们还为缓存指定了一个命名空间，以确保不同的嵌入模型之间不会出现冲突。
embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,  # 实际生成嵌入的工具
    store,  # 嵌入的缓存位置
    namespace=underlying_embeddings.model  # 嵌入缓存的命名空间
)

# 使用embedder为两段文本生成嵌入。
# 结果，即嵌入向量，将被存储在上面定义的内存存储中。
embeddings = embedder.embed_documents(["你好", "智能鲜花客服"])
print(embeddings)

