from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate

# 加载环境变量
load_dotenv()

# 初始化大模型
llm = ChatOpenAI(temperature=0)

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 定义 ReAct 提示模板
template = """你是一个专业的助手，会根据需要使用工具来完成任务。

你可以使用的工具有:
{tools}

使用工具时要遵循以下格式:
Thought: 分析当前需要完成什么，应该如何行动
Action: 要使用的工具名称要采取的行动，必须是[{tool_names}]中的一个
Action Input: 提供给工具的输入
Observation: 工具返回的结果
... (这个思考-行动-观察的循环会继续，直到得到最终答案)
Final Answer: 为用户提供完整的答案

问题: {input}

{agent_scratchpad}"""

# 创建 PromptTemplate 对象
prompt = PromptTemplate(
    template=template,
    input_variables=[
        "tools",
        "tool_names",
        "input",
        "agent_scratchpad",
    ],
)

tool_names = [tool.name for tool in tools]

# 创建 ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,  # 使用 PromptTemplate 对象
)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示详细执行过程
    handle_parsing_errors=True,  # 处理解析错误
)

# 执行查询
try:
    result = agent_executor.invoke(
        {
            "input": "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
        }
    )
    print("\n最终答案:", result["output"])
except Exception as e:
    print(f"执行出错: {e}")
