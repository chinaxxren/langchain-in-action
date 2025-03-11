'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

# ConversationChain 的特点：
# - 它是一个带有记忆功能的对话链
# - 能够记住之前的对话内容
# - 可以进行上下文相关的连续对话
# - 自动维护对话历史

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需的库
from langchain_openai import OpenAI  # 更新导入路径
from langchain.chains import ConversationChain

# 初始化大语言模型
llm = OpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"  # 更新为新的模型名称
)

# 初始化对话链
conv_chain = ConversationChain(llm=llm)

# 打印对话的模板
print(conv_chain.prompt.template)