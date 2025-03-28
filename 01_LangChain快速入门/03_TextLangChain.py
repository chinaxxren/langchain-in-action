
from dotenv import load_dotenv  # 用于加载环境变量
#通过langchain调用openai，不需要设置api_key和base_url，通过环境变量就可以了
from langchain_openai import OpenAI

load_dotenv()  # 加载 .env 文件中的环境变量

llm = OpenAI(  
    model="gpt-3.5-turbo-instruct",
    temperature=0.8,
    max_tokens=60)

# 使用 invoke 方法替代已弃用的 predict 方法
response = llm.invoke("请给我的花店起个名")
print(response.strip())