# SQLDatabaseChain：
# - 简单的统计查询
# - 直接的数据检索
# - 性能要求较高的场景

# 简单直接的查询方式
# - 直接将自然语言转换为 SQL 查询
# - 适合简单、明确的数据库查询
# - 执行流程：
#   自然语言 -> SQL 查询 -> 数据库结果 -> 自然语言回答
   
from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")

# 创建OpenAI实例
llm = OpenAI(temperature=0, verbose=True)

# 创建SQL数据库链实例
db_chain = SQLDatabaseChain(
    llm=llm,
    database=db,
    verbose=True,
    return_direct=True  # 直接返回结果
)

# 运行查询
questions = [
    "有多少种不同的鲜花？",
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
    "从法国进口的鲜花有多少种？",
    "哪种鲜花的销售量最高？"
]

for question in questions:
    print(f"\n问题：{question}")
    response = db_chain.invoke(question)  # 使用 invoke 替代 run
    print(f"回答：{response}")
