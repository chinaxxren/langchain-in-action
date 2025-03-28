
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  # 更新导入路径
from langchain_openai import OpenAI  # 更新导入路径

load_dotenv()

# 原始字符串模板
template = "{flower}的花语是?"
# 创建LangChain模板
prompt_temp = PromptTemplate.from_template(template) 
# 根据模板创建提示
prompt = prompt_temp.format(flower='玫瑰')
# 打印提示的内容
print(prompt)

# 创建模型实例
model = OpenAI(temperature=0)
# 传入提示，调用模型，返回结果
result = model.invoke(prompt)  # 使用 invoke 替代直接调用
print(result)