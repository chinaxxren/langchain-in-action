# 使用 SQL Agent：
# - 复杂的分析问题
# - 需要多步骤推理
# - 需要动态调整查询策略
# - 需要更自然的交互体验

# 智能和灵活的查询方式
#    - 可以进行多步推理和复杂查询
#    - 具有以下特点：
#      - 可以拆分复杂问题
#      - 可以组合多个查询
#      - 可以根据中间结果调整查询策略
#    - 执行流程：
#      思考（分析问题）-> 行动（执行查询）-> 观察（分析结果）-> 继续（根据需要进行下一步）

from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import os

# 连接到FlowerShop数据库
db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
llm = OpenAI(temperature=0, verbose=True)

# 创建SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# 使用Agent执行SQL查询
questions = [
    "哪种鲜花的存货数量最少？",
    "平均销售价格是多少？",
]

for question in questions:
    print(f"\n问题：{question}")
    response = agent_executor.invoke(question)  # 使用 invoke 替代 run
    print(f"回答：{response}")
