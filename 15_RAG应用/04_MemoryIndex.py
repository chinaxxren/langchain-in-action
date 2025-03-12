from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.embeddings import CacheBackedEmbeddings
import os

# 创建内存存储实例
store = InMemoryStore()

# 创建基础 embeddings 实例
underlying_embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 创建带缓存的 embeddings 实例
embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

# 加载文本文件
loader = TextLoader('花语大全.txt', encoding='utf8')

# 创建 LLM 实例
llm = ChatOpenAI(temperature=0)

# 创建索引时使用带缓存的 embeddings
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embeddings,
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
).from_loaders([loader])

# 定义查询字符串, 使用创建的索引执行查询
query = "玫瑰花的花语是什么？"
result = index.query(query, llm=llm)
print(result)

