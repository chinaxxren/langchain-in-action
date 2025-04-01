
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
# 临时移除可能导致问题的环境变量
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']

key = os.environ["OPENAI_API_KEY"]
print(key)
api = os.environ["OPENAI_API_BASE"]
print(api)

from langchain_openai import OpenAI
llm = OpenAI(
    openai_api_key=key,
    openai_api_base=api,
    model_name="gpt-3.5-turbo-instruct", 
    max_tokens=200
)
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)