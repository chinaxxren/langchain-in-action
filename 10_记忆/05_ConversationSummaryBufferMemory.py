#ConversationSummaryBufferMemory 的特点：

# - 结合了缓冲和摘要两种方式
# - 当对话量小时保留完整对话
# - 超过 token 限制时自动生成摘要
# - 保持了良好的上下文连贯性

#解释这两种记忆机制的主要区别：

# 1. 工作方式：
# - ConversationSummaryMemory ：
#   - 直接对所有对话生成摘要
#   - 只保存摘要信息
#   - 每次对话后都会更新摘要
# - ConversationSummaryBufferMemory ：
#   - 结合了缓冲区和摘要两种机制
#   - 在 token 限制内保留完整对话
#   - 超出限制时才生成摘要
# 2. 内存使用：
# - ConversationSummaryMemory ：
#   - 内存使用量稳定
#   - 只存储一份摘要
# - ConversationSummaryBufferMemory ：
#   - 内存使用较大
#   - 同时保存最近对话和摘要
#   - 可以通过 max_token_limit 控制
# 3. 使用场景：
# - ConversationSummaryMemory 适合：
#   - 长期对话但内存受限
#   - 只需要关键信息的场景
# - ConversationSummaryBufferMemory 适合：
#   - 需要更细致的上下文
#   - 短期需要完整对话
#   - 资源充足的环境

# 设置OpenAI API密钥
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需的库
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

# 初始化大语言模型
llm = OpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"  # 更新为新的模型名称
)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=300
    )
)

# 第一天的对话
# 回合1
result = conversation("我姐姐明天要过生日，我需要一束生日花束。")
print(result)

# 回合2
result = conversation("她喜欢粉色玫瑰，颜色是粉色的。")
# print("\n第二次对话后的记忆:\n", conversation.memory.buffer)
print(result)

# 第二天的对话
# 回合3
result = conversation("我又来了，还记得我昨天为什么要来买花吗？")
print(result)