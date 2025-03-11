#1. 滑动窗口记忆机制：
# ```python
# memory=ConversationBufferWindowMemory(k=1)
#  ```
# - k=1 表示只保留最近的 1 轮对话
# - 像滑动窗口一样，新对话进来时，最老的对话会被移出
# - 比 ConversationBufferMemory 更节省内存

#- ConversationBufferMemory ：
# - 存储所有历史对话
# - 不会自动删除任何对话记录
# - 记忆会持续累积
# - ConversationBufferWindowMemory ：
# - 只保留最近 k 轮对话
# - 使用滑动窗口机制
# - 超出窗口的旧对话会被自动删除

#加载环境
from dotenv import load_dotenv
load_dotenv()

# 导入所需的库
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# 初始化大语言模型
llm = OpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"  # 更新为新的模型名称
)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=1)
)

# 第一天的对话
# 回合1
result = conversation.invoke("我姐姐明天要过生日，我需要一束生日花束。")  # 使用 invoke 方法
print(result)
# 回合2
result = conversation.invoke("她喜欢粉色玫瑰，颜色是粉色的。")  # 使用 invoke 方法
print(result)

# 第二天的对话
# 回合3
result = conversation.invoke("我又来了，还记得我昨天为什么要来买花吗？")  # 使用 invoke 方法
print(result)
