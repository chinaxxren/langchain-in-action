'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

from dotenv import load_dotenv
load_dotenv()

#----第一步 创建提示
# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate  # 更新导入路径
# 原始字符串模板
template = "{flower}的花语是?"
# 创建LangChain模板
prompt_temp = PromptTemplate.from_template(template) 
# 根据模板创建提示
prompt = prompt_temp.format(flower='玫瑰')
# 打印提示的内容
print(prompt)

#----第二步 创建并调用模型 
# 导入LangChain中的OpenAI模型接口
from langchain_openai import OpenAI  # 更新导入路径
# 创建模型实例
model = OpenAI(temperature=0)
# 传入提示，调用模型，返回结果
result = model.invoke(prompt)  # 使用 invoke 替代直接调用
print(result)