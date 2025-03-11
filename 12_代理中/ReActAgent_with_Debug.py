'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI和SERPAPI的API密钥
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 试一试LangChain的Debug和Verbose，看看有何区别
import langchain
# langchain.debug = True   # 详细的调试信息
langchain.verbose = True   # 基本的运行信息

# 配置日志输出
import logging
import sys
# - 设置日志输出到标准输出（控制台）
# - 设置日志级别为 DEBUG（最详细）
# - 添加额外的处理器确保日志正确输出
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# 加载所需的库
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# 初始化大模型
llm = OpenAI(temperature=0)

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True) #verbose=True 让我们能看到 Agent 的决策过程
- 

# 跑起来
agent.run("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")