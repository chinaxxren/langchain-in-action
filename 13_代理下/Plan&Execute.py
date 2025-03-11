#解释一下这个 Plan and Execute Agent 的执行过程：

# 1. 计划制定（Planning）：
#    Agent 制定了四个步骤的计划：
# - 确定纽约玫瑰花束的平均价格
# - 用100美元除以平均价格计算可买数量
# - 将结果向下取整（因为不能买半束花）
# - 提供最终答案

# 2. 执行过程（Execution）：
#    第一步：搜索价格
# - 使用 Search 工具获取价格信息
# - 得到价格范围：$10（街边摊贩）到 $120（高端花店）
# - Agent 选择了最低价格 $10 作为计算基准

# 第二步：计算数量
# - 使用 Calculator 工具计算：100 ÷ 10 = 10
# - 得出可以买 10 束玫瑰

# 第三步：取整
# - 因为结果已经是整数，不需要额外取整操作

# 第四步：总结
# - 给出最终答案：可以买 10 束玫瑰（按最低价格计算）

from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain

search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

#- 定义了两个工具：搜索和计算器
# - 每个工具都有名称、功能和描述
# - Agent 会根据描述选择合适的工具
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]
model = ChatOpenAI(temperature=0)

# 规划组件
planner = load_chat_planner(model)
# 执行组件
executor = load_agent_executor(model, tools, verbose=True)
# 初始化 Agent，组合
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 使用 invoke 方法替代 run
# 发现价格范围：$10（街边摊贩）到 $120（高端花店）
agent.invoke("在纽约，100美元能最多买几束玫瑰?")

# - Planner：负责制定执行计划
# - Executor：负责执行具体任务
# - 两者协同工作完成复杂任务
# 3. 特点：
# - 分步执行：先规划后执行
# - 可追踪：verbose=True 显示详细过程
# - 灵活性：可以处理复杂的多步骤任务
# 4. 优势：
# - 结构化的任务分解
# - 清晰的执行流程
# - 便于调试和优化