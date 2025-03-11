#解释这两种 Agent 类型的区别：

# 1. SELF_ASK_WITH_SEARCH ：
# - 自问自答式的代理
# - 工作流程：
#   - 将复杂问题分解成子问题
#   - 逐个搜索解答子问题
#   - 整合答案得出最终结果
# - 适用场景：
#   - 需要多步推理的复杂问题
#   - 问题可以分解成多个子问题
#   - 每个子问题都需要搜索
# 2. ZERO_SHOT_REACT_DESCRIPTION ：
# - 基于思考-行动-观察的代理
# - 工作流程：
#   - 思考（Thought）：分析当前情况
#   - 行动（Action）：选择合适的工具
#   - 观察（Observation）：获取结果
# - 适用场景：
#   - 直接的问答任务
#   - 需要工具组合的任务
#   - 单步或多步执行的任务
# 3. 主要区别：
# - 问题处理方式：
#   - SELF_ASK：自动分解问题
#   - ZERO_SHOT：直接选择工具
# - 灵活性：
#   - SELF_ASK：更适合复杂推理
#   - ZERO_SHOT：更通用和稳定

from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer", 
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.SELF_ASK_WITH_SEARCH, 
    verbose=True,
    handle_parsing_errors=True
)

# 修改问题表述，使其更容易被分解
self_ask_with_search.invoke(
    "哪个国家把牡丹作为国花？这个国家的首都是哪里？"  # 将一个复杂问题分解为两个简单问题
)

# > Entering new AgentExecutor chain...
# Could not parse output:  是的。
# Followup: 这个国家的首都是什么？

# Intermediate answer: Invalid or incomplete response
# Could not parse output: Followup: Could you please provide the country in English?

# Intermediate answer: Invalid or incomplete response
# Could not parse output: Followup: Which country has peony as its national flower?
# Intermediate answer: China
# Followup:What is the capital of China?
# Intermediate answer: Beijing
# So the final answer is: China

# > Finished chain.