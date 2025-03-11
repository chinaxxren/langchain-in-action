'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os  # 添加 os 模块导入

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  temperature=0.5,
  max_tokens=100,
  prompt="请给我的花店起个名")

# strip() 方法用于去除文本前后的空白字符（包括空格、换行符、制表符等）
print(response.choices[0].text.strip())