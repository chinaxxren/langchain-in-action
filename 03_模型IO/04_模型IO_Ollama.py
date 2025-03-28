from dotenv import load_dotenv  # 用于加载环境变量
# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

load_dotenv()  # 加载 .env 文件中的环境变量

# 创建 Ollama 实例
# 默认连接到 http://localhost:11434
llm = Ollama(model="llama3.2")  # 或其他已下载的模型名称

# 创建提示模板
template = """You are a flower shop assistant.
For {price} of {flower_name}, can you write something for me?
"""
prompt = PromptTemplate.from_template(template)

# 格式化提示并获取响应
formatted_prompt = prompt.format(flower_name="玫瑰", price="50")
response = llm.invoke(formatted_prompt)
print(response)
