from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

# 创建模型实例
llm = OpenAI(temperature=0)

# 创建提示模板
prompt = PromptTemplate.from_template("{flower}的花语是?")

# 使用 | 运算符创建管道
chain = prompt | llm

# 调用链，使用字典参数
result = chain.invoke({"flower": "玫瑰"})

# 打印结果
print(result)