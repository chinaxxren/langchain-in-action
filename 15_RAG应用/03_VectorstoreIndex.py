from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import os

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 加载文本文件
loader = TextLoader('花语大全.txt', encoding='utf8')

# 创建 LLM 实例
llm = ChatOpenAI(temperature=0)

# 创建索引时指定 Chroma 和其他参数
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embeddings,
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
).from_loaders([loader])

# 定义查询字符串, 使用创建的索引执行查询
query = "玫瑰花的花语是什么？"
result = index.query(query, llm=llm)  # 添加 llm 参数
print(result)

