#STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 这个代理类型：
# 这是 LangChain 中一个特殊的代理类型，它结合了几个重要特性：

# 1. STRUCTURED_CHAT （结构化对话）：
#    - 使用结构化格式进行交互
#    - 更擅长遵循特定的格式和协议
#    - 输出更加规范和可预测

# 2. ZERO_SHOT （零样本）：   
#    - 无需预先训练的示例
#    - 可以直接根据任务描述做出决策
#    - 适应性强，可处理新任务

# 3. REACT （推理和行动框架）：   
#    - 思考（Reasoning）：分析需要做什么
#    - 行动（Acting）：执行具体操作
#    - 观察（Observing）：查看结果
#    - 继续（Continuing）：基于观察继续推理

# 4. DESCRIPTION （描述式）：
#    - 通过自然语言描述来理解工具功能
#    - 更容易理解和使用各种工具
# 在您的代码中，这个代理类型特别适合：

# - 网页内容分析
# - 结构化数据提取
# - 多步骤推理任务
# - 需要使用多个工具的复杂任务

from dotenv import load_dotenv
load_dotenv()

# 更新导入路径
from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# 创建异步浏览器实例
async_browser = create_async_playwright_browser()
#- 工具集（PlayWright浏览器工具）
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

# LLM不稳定，对于这个任务，可能要多跑几次才能得到正确结果
llm = ChatOpenAI(temperature=0.5)  

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

async def main():
    query = """
    Please navigate to python.langchain.com and analyze the page structure:
    1. First, extract all h1-h6 heading elements
    2. For each heading, get its text content and level
    3. Return a structured list of headings with their hierarchy
    Note: Focus only on HTML heading elements to avoid non-HTML node errors
    """
    response = await agent_chain.ainvoke(query)
    
    try:
        # 优化输出格式
        headers = response['output']
        print("\n=== 网页标题结构 ===")
        if isinstance(headers, list):
            for i, header in enumerate(headers, 1):
                print(f"{i}. {header}")
        else:
            print(headers)
        print("==================")
    except Exception as e:
        print(f"处理响应时出错: {str(e)}")

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())