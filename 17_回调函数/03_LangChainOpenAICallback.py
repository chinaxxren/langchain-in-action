# 设置OpenAI和SERPAPI的API密钥
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import asyncio
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback  # 更新导入路径

# 初始化大语言模型
llm = OpenAI(temperature=0.5)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# 使用context manager进行token counting
# with 上下文管理器() as 变量名:
# 上下文管理器的工作原理：
# - 进入 with 块时自动调用 __enter__ 方法
# - 退出 with 块时自动调用 __exit__ 方法
# - 确保资源的正确释放和清理
# 主要优点：
# - 自动管理资源
# - 确保代码执行完后进行清理
# - 简化错误处理
# - 使代码更简洁和安全
# 在 get_openai_callback() 的例子中，它用于：
# - 自动开始统计 API 调用
# - 自动结束统计
# - 确保统计数据的准确性
with get_openai_callback() as cb:
    # 第一天的对话
    # 回合1
    conversation.invoke("我姐姐明天要过生日，我需要一束生日花束。") 
    print("第一次对话后的记忆:", conversation.memory.buffer)

    # 回合2
    conversation.invoke("她喜欢粉色玫瑰，颜色是粉色的。")
    print("第二次对话后的记忆:", conversation.memory.buffer)

    # 回合3 （第二天的对话）
    conversation.invoke("我又来了，还记得我昨天为什么要来买花吗？")
    print("/n第三次对话后时提示:/n",conversation.prompt.template)
    print("/n第三次对话后的记忆:/n", conversation.memory.buffer)

# 输出使用的tokens
print("\n总计使用的tokens:", cb.total_tokens)

# 进行更多的异步交互和token计数
async def additional_interactions():
    with get_openai_callback() as cb:
        #异步并发请求：
        # - 使用 asyncio.gather 同时发送多个请求
        # - 并行处理 3 个相同的问题
        # - 提高处理效率，减少等待时间
        await asyncio.gather(
            *[llm.agenerate(["我姐姐喜欢什么颜色的花？"]) for _ in range(3)]
        )
    print("\n另外的交互中使用的tokens:", cb.total_tokens)

# 运行异步函数
asyncio.run(additional_interactions())
