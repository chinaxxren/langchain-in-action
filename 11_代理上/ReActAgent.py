#让 Agent 能够：
# - 自主决定使用哪个工具
# - 按照逻辑顺序完成任务
# - 组合多个工具的结果

# 设置OpenAI和SERPAPI的API密钥
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 加载所需的库
from langchain_community.agent_toolkits.load_tools import load_tools  # 更新导入路径
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI  # 更新导入路径

# 初始化大模型
#4. temperature=0：
# - 设置为0表示输出最确定性的结果
# - 适合需要准确计算的场景
llm = OpenAI(temperature=0)

# 设置工具
#- serpapi：用于搜索网络获取玫瑰花价格信息
#- llm-math：用于计算加价15%后的价格
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent
agent = initialize_agent(
    tools,
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #Agent 会按照 "思考-行动-观察" 的方式工作
    verbose=True)

# 跑起来
#3. 工作流程：
# - Agent 收到问题后会：
#   1. 先使用 serpapi 搜索玫瑰花的市场价格
#   2. 获得价格后，使用 llm-math 计算加价15%的结果
#   3. 最后整合信息给出完整答案
agent.run("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")