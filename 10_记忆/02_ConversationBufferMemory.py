#1. ConversationBufferMemory 的特点：
# - 这是一个简单的对话记忆缓冲器
# - 按时间顺序存储所有对话历史
# - 包括人类输入和 AI 的回答
# - 不会遗忘或压缩任何内容
# - 每次对话都会被存储在 memory.buffer 中
# - 新的对话会自动获取之前的上下文
# - AI 可以根据历史记录理解上下文

from dotenv import load_dotenv
load_dotenv()

# 导入所需的库
from langchain_openai import OpenAI  # 更新导入路径
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# 初始化大语言模型
llm = OpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"  # 更新为新的模型名称
)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    # 设置记忆容量，只保留最近的2次对话
    memory=ConversationBufferMemory(k=2)
    # 或者通过令牌数限制：memory=ConversationBufferMemory(max_token_limit=100)
)

# 第一天的对话
# 回合1
conversation.invoke("我姐姐明天要过生日，我需要一束生日花束。")  # 使用 invoke 方法
print("第一次对话后的记忆:", conversation.memory.buffer)

# 回合2
conversation.invoke("她喜欢粉色玫瑰，颜色是粉色的。")  # 使用 invoke 方法
print("第二次对话后的记忆:", conversation.memory.buffer)

# 回合3 （第二天的对话）
conversation.invoke("我又来了，还记得我昨天为什么要来买花吗？")  # 使用 invoke 方法
print("\n第三次对话后时提示:\n", conversation.prompt.template)
print("\n第三次对话后的记忆:\n", conversation.memory.buffer)

# 第三次对话后时提示:
# The following is a friendly conversation between a human and an AI. 
# The AI is talkative and provides lots of specific details from its 
# context. If the AI does not know the answer to a question, it
#  truthfully says it does not know.

# Current conversation:
# {history}
# Human: {input}
# AI:
