#1. 摘要记忆机制：
# ```python
# memory=ConversationSummaryMemory(llm=llm)
#  ```

# - 不是存储完整的对话历史
# - 而是使用 LLM 对对话内容进行总结
# - 保存的是对话的摘要信息

# 设置OpenAI API密钥
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需的库
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory

# 初始化大语言模型
llm = OpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"  # 更新为新的模型名称
)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

# 第一天的对话
# 回合1：开始记录对话
result = conversation.invoke("我姐姐明天要过生日，我需要一束生日花束。")  # 使用 invoke 方法
print(result)

# 回合2：更新摘要
result = conversation.invoke("她喜欢粉色玫瑰，颜色是粉色的。")  # 使用 invoke 方法
print(result)

# 第二天的对话
# 回合3：基于摘要回答
result = conversation.invoke("我又来了，还记得我昨天为什么要来买花吗？")  # 使用 invoke 方法
print(result)